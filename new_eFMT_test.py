import torch
import cv2
import torch_utils as tu
from torch_CC_model import Network
from torch_tranining import transform


if __name__ == "__main__":
    device = "cpu"
    img_o = torch.tensor(cv2.imread("shangai images/DJI_20201217154322_0100_W.JPG", 1), dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)[:, :, 2:-2, 2:-2]
    temp_o = torch.tensor(cv2.imread("shangai images/DJI_20201217154324_0101_W.JPG", 1), dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)[:, :, 2:-2, 2:-2]
    
    N = Network(argmax_beta=1000.0, device=device).to(device)
    T = transform(temp_o, R = 0, S=3, X=-1000, Y=-1000, device=device)
    res, cr1, cr2 = N(img_o, T, True)

    
