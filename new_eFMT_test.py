import torch
import cv2
import torch_utils as tu
from torch_CC_model import Network
from torch_tranining import transform


if __name__ == "__main__":
    device = "cpu"
    img_o = torch.tensor(cv2.imread("T_S02951.png", 1), dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)[:, :, 2:-2, 2:-2]
    temp_o = torch.tensor(cv2.imread("T_S02952.png", 1), dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)[:, :, 2:-2, 2:-2]
    
    N = Network(argmax_beta=1000.0, device=device).to(device)
    I = transform(img_o, R = 0, S=3, X=-200, Y=-400, device=device)
    T = transform(temp_o, R = 0, S=3, X=-200, Y=-400, device=device)
    res, cr1, cr2 = N(I, T, True)

    
