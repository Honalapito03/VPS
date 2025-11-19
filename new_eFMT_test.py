import torch
import cv2
import torch_utils as tu
from torch_CC_model import Network
from torch_tranining import transform


import numpy as np
import cv2

def hann_window(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)

def pad_to_center(img, target_shape):
    """Center-pad img inside an all-zero target canvas."""
    H, W, C = target_shape
    h, w, c = img.shape

    out = np.zeros((H, W, C), dtype=img.dtype)
    cy, cx = H//2, W//2
    y0, x0 = cy - h//2, cx - w//2
    out[y0:y0+h, x0:x0+w, :] = img
    return out

import numpy as np

def gaussian_mask(h, w, center=None, sigma=50, vmin=0.0, vmax=1.0):
    """
    Create a 2D Gaussian mask with controllable minimum and maximum.
    
    h, w     : height and width of mask
    center   : (cx, cy) optional; if None, mask is centered
    sigma    : standard deviation of Gaussian
    vmin     : minimum mask value
    vmax     : maximum mask value
    """

    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    # Coordinate grid
    y, x = np.ogrid[:h, :w]

    # Standard Gaussian (range 0..1)
    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2*sigma*sigma))

    # Scale to [vmin, vmax]
    mask = vmin + (vmax - vmin) * gaussian

    return mask


if __name__ == "__main__":
    device = "cpu"
    img_o = cv2.imread("ShanghaiTech_Campus/Images/DJI_20201217154322_0100_W.JPG", 1)   
    print("Original image shape:", img_o.shape)

    # Suppose the small object is at (x=450, y=300, w=20, h=20)

    temp = cv2.imread("ShanghaiTech_Campus/Images/DJI_20201217154324_0101_W.JPG", 1) * gaussian_mask(img_o.shape[0], img_o.shape[1], center=(2500, 1600), sigma=100, vmin=0.1, vmax=1.0)[:,:,np.newaxis]

    
    img_o = torch.tensor(img_o, dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)

    N = Network(argmax_beta=300.0, device=device).to(device)
    T = torch.tensor(temp, dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)
    res, cr1, cr2 = N(img_o, T, True)

    "HELOOOOOO"
    
