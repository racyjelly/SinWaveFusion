
import torch
import torchvision.transforms as transforms
from torchvision import utils
from SinWaveFusion.DWT_IDWT_layer import DWT_2D, IDWT_2D

from skimage import morphology, filters
from inspect import isfunction
import numpy as np
from PIL import Image
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False




def dilate_mask(mask, mode):
    if mode == "harmonization":
        element = morphology.disk(radius=7)
    if mode == "editing":
        element = morphology.disk(radius=20)
    mask = mask.permute((1, 2, 0))
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, footprint=element)
    mask = filters.gaussian(mask, sigma=5)
    mask = mask[:, :, None, None]
    mask = mask.transpose(3, 2, 0, 1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


# for roi_sampling

def stat_from_bbs(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    bb_mean = torch.mean(image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    bb_std = torch.std(image[:, :, y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    return [bb_mean, bb_std]


def extract_patch(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    image_patch = image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb]
    return image_patch


# for clip sampling
def thresholded_grad(grad, quantile=0.8):
    """
    Receives the calculated CLIP gradients and outputs the soft-tresholded gradients based on the given quantization.
    Also outputs the mask that corresponds to remaining gradients positions.
    """
    grad_energy = torch.norm(grad, dim=1)
    grad_energy_reshape = torch.reshape(grad_energy, (grad_energy.shape[0],-1))
    enery_quant = torch.quantile(grad_energy_reshape, q=quantile, dim=1, keepdim=False, interpolation='nearest')[:,None,None] #[batch ,1 ,1]
    gead_energy_minus_energy_quant = grad_energy - enery_quant
    grad_mask = (gead_energy_minus_energy_quant > 0)[:,None,:,:]

    gead_energy_minus_energy_quant_clamp = torch.clamp(gead_energy_minus_energy_quant, min=0)[:,None,:,:]#[b,1,h,w]
    unit_grad_energy = grad / grad_energy[:,None,:,:] #[b,c,h,w]
    unit_grad_energy[torch.isnan(unit_grad_energy)] = 0
    sparse_grad = gead_energy_minus_energy_quant_clamp * unit_grad_energy #[b,c,h,w]
    return sparse_grad, grad_mask

# helper functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def img_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    # 0-1
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def img_denormalize(normalized_tensor, min_val, max_val): 
    denormalized = (normalized_tensor) * (max_val - min_val) + min_val
    return denormalized


def create_img_scales(foldername, filename, scale_factor=1.411, image_size=None, create=False, auto_scale=None, set_device=None):
    
    orig_image = Image.open(foldername + filename)

    if orig_image.mode == 'RGBA':
        # 흰색 배경에 이미지를 합성
        background = Image.new('RGB', orig_image.size, (255, 255, 255))
        background.paste(orig_image, mask=orig_image.split()[3])
        orig_image = background
        
    width, height= orig_image.size

    # convert to PNG extension for lossless conversion
    filename = filename.rsplit( ".", 1 )[ 0 ] + '.png'
    if image_size is None:
        image_size = (orig_image.size)
    if auto_scale is not None:
        scaler = np.sqrt((image_size[0] * image_size[1])/auto_scale)
        if scaler > 1:
            image_size = (int(image_size[0]/scaler), int(image_size[1]/scaler))
    sizes = []
    downscaled_images = []
    recon_images = []
    rescale_losses = []
    tensor_list = []

    if set_device is None:
        set_device = "cuda:2" if torch.cuda.is_available else "cpu"

    # auto resize
    # rf_net = 35
    area_scale_0 = 3110  # defined such that rf_net^2/area_scale0 ~= 40%
    s_dim = min(image_size[0], image_size[1])
    l_dim = max(image_size[0], image_size[1])
    scale_0_dim = int(round(np.sqrt(area_scale_0*s_dim/l_dim)))
    # clamp between 42 and 55
    scale_0_dim = 42 if scale_0_dim < 42 else (55 if scale_0_dim > 55 else scale_0_dim )
    small_val = scale_0_dim
    min_val_image = min(image_size[0], image_size[1])
    n_scales = int((round( (np.log(min_val_image/small_val) + 1) / (np.log(scale_factor) + 1 ) )))
    #n_scales = int(round( (np.log(min_val_image/small_val)) / (np.log(scale_factor)) ) -1)
    scale_factor = np.exp((np.log(min_val_image / small_val)) / (n_scales - 1))
    dwt = DWT_2D("haar")
    idwt = IDWT_2D("haar")
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    transform_to_image = transforms.Compose([transforms.ToPILImage()])

    for i in range(n_scales):
        if width > 400 and height > 300:
            after_width = width if width % 16 == 0 else (width // 16) * 16
            after_height = height if height % 16 ==0 else (height // 16) * 16

            cur_img = orig_image.resize((after_width, after_height), Image.LANCZOS)
        else:
            after_width = width if width % 8 == 0 else (width // 8) * 8
            after_height = height if height % 8 ==0 else (height // 8) * 8

            cur_img = orig_image.resize((after_width, after_height), Image.LANCZOS)

        if i==0:
            img_tensor = transform_to_tensor(cur_img).to(set_device)
            img_tensor = img_tensor.unsqueeze(0)
            # Wavelet transform
            ll, lh, hl, hh = dwt(img_tensor)
            ll_save1 = ll.squeeze(0)
            path_to_save = foldername + 'scale_' + str(i) + '/'
            if create:
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                transform_to_image(ll_save1).save(path_to_save + filename)
            
            downscaled_images.append(transform_to_image(ll_save1))
            tensor_list.append(ll_save1)
            sizes.append(transform_to_image(ll_save1).size)
        
        else:
            cur_img = cur_img
            path_to_save = foldername + 'scale_' + str(i) + '/'
            if create:
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                cur_img.save(path_to_save + filename)
            
            downscaled_images.append(cur_img)
            tensor_list.append(transform_to_tensor(cur_img))
            sizes.append(cur_img.size)

    for i in range(n_scales - 1):
        recon_image = downscaled_images[i].resize(sizes[i + 1], Image.BILINEAR)
        rescale_losses.append(
                np.linalg.norm(np.subtract(downscaled_images[i + 1], recon_image)) / np.asarray(recon_image).size)
        
    return sizes, rescale_losses, scale_factor, n_scales, tensor_list

