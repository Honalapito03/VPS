import torch
import cv2
import torch_utils as tu
from torch_CC_model import Network
from torch_tranining import transform

if __name__ == "__main__":
    device = "cpu"
    img_o = torch.tensor(cv2.imread("horse0.jpg", 1), dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)[:, :, 2:-2, 2:-2]

    N = Network(argmax_beta=300.0, device=device).to(device)
    T = transform(img_o, R = -170, S=1, X=100, Y=100, device=device)
    res, cr1, cr2 = N(img_o, T, True)

    "HELOOOOOO"
    
