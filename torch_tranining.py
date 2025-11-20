import torch
import numpy as np
import cv2
import torch
from torch.distributions import Uniform

import torch_utils as tu
from torch_CC_model import Network

def one_batch(img_o, template_o, N,
            R = 0,
            S = 1,
            X = 0,
            Y = 0,
            device="cuda"):
    if R is 0 : R = torch.zeros((img_o.size()[0]), device=device)
    if S is 1 : S = torch.ones((img_o.size()[0]), device=device)

    if X is 0 : X = torch.zeros((img_o.size()[0]), device=device)
    if Y is 0 : Y = torch.zeros((img_o.size()[0]), device=device)

    img = img_o
    temp = template_o

    cr1_s, cr2_s = N.calc_out_size(list(img.shape))
    cr1_s[1] = 1  #the map is one-channel
    cr2_s[1] = 1  #the map is one-channel

    rotation_cord = torch.round(R * cr1_s[2] / 360 + cr1_s[2] // 2)

    cy, cx = (img.shape[3] / 2, img.shape[2] / 2)
    r = torch.log(torch.sqrt(torch.tensor(cy**2 + cx**2, device=device)))
    scale_cord = torch.round(torch.log(torch.tensor(S)) / r * cr1_s[3] + cr1_s[3] // 2)

    #rotation_cord, scale_cord = torch.meshgrid(rotation_cord, scale_cord, indexing='ij')

    R_S_cord = torch.stack([torch.arange(0, rotation_cord.size()[0], device=device), torch.zeros_like(rotation_cord, device=device), scale_cord, rotation_cord], 1).long()


    #rotation-scale true map
    R_S = torch.zeros(tuple([int(x.item()) for x in cr1_s]), device=device)
    R_S[R_S_cord[:, 0], R_S_cord[:, 1], R_S_cord[:, 2], R_S_cord[:, 3]] = 1

    #translation true map
    X_Y_cord = torch.stack([torch.arange(0, X.size()[0], device=device), torch.zeros_like(X, device=device), X, Y], 1).long()
    X_Y = torch.zeros(tuple([int(x.item()) for x in cr2_s]), device=device)
    X_Y[X_Y_cord[:, 0], X_Y_cord[:, 1], X_Y_cord[:, 2], X_Y_cord[:, 3]] = 1
    #print("rota: ", rotation_cord / cr1_s[2] * 360)
    #print("coord: ", R_S_cord)
    #print()
    #imshow("rota", R_S[0].detach().transpose(0, 2).transpose(0, 1).cpu().numpy())
    #imshow("trans", X_Y[0].detach().transpose(0, 2).transpose(0, 1).cpu().numpy())


    #results of the network
    res, cr1, cr2 = N(img, temp, True)
    error = ((R_S - cr1)**2).mean() + ((X_Y - cr2)**2).mean()
    return(res, error)


def transform(  img_o,
                R = 0,
                S = 1,
                X = 0,
                Y = 0,
             device ="cuda"):
    #IMPORTANT: First translates, then rotates and scales from the midpoint
    R = torch.tensor([R])
    S = torch.tensor([S])
    X = torch.tensor([X])
    Y = torch.tensor([Y])
    m = torch.zeros((2, 3), device=device)
    m[0, 2] = -X
    m[1, 2] = -Y
    m[0, 0] = torch.cos(R * np.pi / 180) * 1 / S
    m[0, 1] = -1 *torch.sin(R * np.pi / 180) * 1 / S
    m[1, 0] = torch.sin(R * np.pi / 180) * 1 / S
    m[1, 1] = torch.cos(R * np.pi / 180) * 1 / S

    return(tu.warp(img_o, m, (img_o.size()[3], img_o.size()[2])))


def training(img_o, batch_num, steps_in_epoch, epoch, device="cuda"):
    N = Network(argmax_beta=100.0, device=device).to(device)

    optimizer = torch.optim.Adam(N.parameters(), lr=0.0001)
    R_dist = Uniform(torch.ones((batch_num), device=device) * 40, torch.ones((batch_num), device=device) * 60) #rotation,
    S_dist = Uniform(torch.ones((batch_num), device=device) * 1, torch.ones((batch_num), device=device) * 1.0001) #scaling
    X_dist = Uniform(torch.ones((batch_num), device=device) * -50, torch.ones((batch_num), device=device) * 50) #X (full size is 701x701)
    Y_dist = Uniform(torch.ones((batch_num), device=device) * -50, torch.ones((batch_num), device=device) * 50) #Y (full size is 701x701)


    R = R_dist.sample()
    S = S_dist.sample()
    X = X_dist.sample() * 0
    Y = Y_dist.sample() * 0
    print(R)
    for E in range(epoch):
        for e in range(steps_in_epoch):

            templates = torch.concat([transform(img_o, R = R[b], S = S[b], X = X[b], Y = Y[b], device=device) for b in range(batch_num)], 0)
            imgs = torch.concat([img_o for b in range(batch_num)], 0)

            res, error = one_batch(imgs, templates, N, R = R, S = S, X = X, Y = Y, device=device)
            #print(res[:, 0])
            optimizer.zero_grad()
            error.backward()
            save = N.convC.real_conv.weight.clone()
            optimizer.step()
            print(e, end=" ")
        print(res[:, 0], R)
        print()
        print(E, error.item())
        print("Rotational error: ", round((R + res[:, 0]).mean().item(), 3))
        print("Scaling error: ", round((S + res[:, 1]).mean().item(), 3))
        print("Translation error: ", round((((X + res[:, 2]) **2 + (X + res[:, 3])**2)**0.5).mean().item(), 3))
        #print(N.convC.real_conv.weight.shape)
        #print((N.convC.real_conv.weight - save) * torch.sign(N.convC.real_conv.weight))
        print()


def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    device = "cpu"
    img_o = torch.tensor(cv2.imread("horse0.jpg", 1), dtype=torch.float32, device=device).transpose(0, 2).transpose(1, 2).unsqueeze(0)[:, :, 2:-2, 2:-2]

    N = Network(argmax_beta=300.0, device=device).to(device)
    T = transform(img_o, R = -120, S=1.3, device=device)
    res, error = one_batch(img_o, T, N, R = torch.tensor([45]), device=device)
    print("Test: ", res)
    #training(img_o, 1, 4, 100, device=device)

    #TODO: Scale overshoot issue