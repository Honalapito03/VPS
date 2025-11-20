import torch
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
import gc
import torch_utils as tu

class Network(torch.nn.Module):
    def __init__(self, resolution=(720,720), argmax_beta=100.0, device="cuda"):
        super().__init__()
        self.plot = False
        self.resolution = resolution
        self.argmax_beta = argmax_beta
        self.device = device

        # D and C have to have the same out dim as A's in dim
        self.convA = tu.ComplexIdentity() #ComplexConv2d(3, 1, 3,  stride=1)
        self.convB = tu.ComplexIdentity() #ComplexConv2d(3, 3, 5, padding=0)
        self.convC = tu.ComplexIdentity() #ComplexConv2d(3, 1, 1, stride=1)
        self.convD = tu.ComplexIdentity() #ComplexConv2d(3, 1, 1)


    def calc_out_size(self, in_size):
        cr1_size = self.convA.calc_out_size(self.convC.calc_out_size(in_size[:2] + list(self.resolution)))
        cr2_size = self.convA.calc_out_size(self.convD.calc_out_size(in_size))
        return(cr1_size, cr2_size)

    def _CPSD(self, a, b):
        return (a * torch.conj(b)) / (torch.abs(a) * torch.abs(b) + 1e-15)  # Add a small constant to avoid division by zero

    def phase_correlation(self, A, B):

        A = fft.fft2(A)
        #for b in range(A.size()[0]):
            #for c in range(A.size()[1]):
                #A[b][c]=fft.fftshift(A[b][c])

        B = fft.fft2(B)
        #for b in range(B.size()[0]):
            #for c in range(B.size()[1]):
                #B[b][c]=fft.fftshift(B[b][c])

        #B[:, :, B.shape[2] // 2 - 1: B.shape[2] // 2 + 1, B.shape[3] // 2 - 1: B.shape[3] // 2 + 1] *= 0

        cpsd = self._CPSD(A, B)
        res = fft.ifft2(cpsd)
        for b in range(res.size()[0]):
            for c in range(res.size()[1]):
                res[b][c]=fft.fftshift(res[b][c])
        return res

    def calculate_shift(self, img, template):
        #CONV A
        print("mean", torch.mean(img))
        img = self.convA(img - torch.mean(img))
        print(torch.mean(img - torch.mean(img)))

        template = self.convA(template)
        cr = self.phase_correlation(img - torch.mean(img), template - torch.mean(template))  # (1,1,H,W)
        res = tu.soft_argmax_2d(cr.real, beta=self.argmax_beta)
        print("result: ", res, cr.shape)
        if self.plot:
          for b in range(img.size()[0]):
              tu.imshow("phase corr", cr[b][0].real.detach().cpu().numpy())
              pass
        return( (res - torch.tensor([cr.shape[2] // 2, cr.shape[3] // 2], device=self.device)) / torch.tensor([cr.shape[2], cr.shape[3]], device=self.device) # shift to center
              , cr.real)

    def get_rotation_scale(self, img, template):
        #CONV B
        img = self.convB(img)
        template = self.convB(template)


        im = fft.fft2(img.real )#torch.complex(img.real, torch.img.real.flip(2).flip() 3)))
        te = fft.fft2(template.real)#torch.complex(template.real, template.real.transpose(2, 3)))
        for b in range(im.size()[0]):
            for c in range(im.size()[1]):
                im[b][c] = fft.fftshift(im[b][c]).abs()
        for b in range(te.size()[0]):
            for c in range(te.size()[1]):
                te[b][c] = fft.fftshift(te[b][c]).abs()

        tu.imshow("log rota", (im.real/torch.max(im.real))[0].transpose(0, 2).transpose(0, 1).detach().cpu().numpy() * 255)
        tu.imshow("log rota temp", (te.real/torch.max(te.real))[0].transpose(0, 2).transpose(0, 1).detach().cpu().numpy() * 255)

        imf = tu.log_polar_transform_torch(im.real, output_shape=self.resolution, device=self.device)
        tempF = tu.log_polar_transform_torch(te.real, output_shape=self.resolution, device=self.device)
        
        
        imf *= (torch.linspace(0, 5, imf.size()[2])**5).expand(imf.size()[0], imf.size()[1], imf.size()[2], imf.size()[3]).to(self.device).transpose(2, 3)
        tempF *= (torch.linspace(0, 5, tempF.size()[2])**5).expand(tempF.size()[0], tempF.size()[1], tempF.size()[2], tempF.size()[3]).to(self.device).transpose(2, 3)

        if self.plot:
            tu.imshow("log polar", (imf.real/torch.max(imf.real))[0].transpose(0, 2).transpose(0, 1).detach().cpu().numpy())
            tu.imshow("log polar temp", (tempF.real/torch.max(tempF.real))[0].transpose(0, 2).transpose(0, 1).detach().cpu().numpy())

        #CONV C
        res, cr = self.calculate_shift(self.convC(imf), self.convC(tempF))
        #print("real cord:", res  * torch.tensor([cr.shape[2], cr.shape[3]], device=self.device) + torch.tensor([cr.shape[2] // 2, cr.shape[3] // 2], device=self.device))
        rotation = -1 * res[:, 1] * 360 # in degrees
        cy, cx = (img.shape[3] / 2, img.shape[2] / 2)
        r = torch.log(torch.tensor(max(cy / 2, cx / 2), device=self.device))
        scale = torch.pow(torch.e, r * res[:, 0])

        #true direction
        rotation_matrix = torch.zeros((img.size()[0], 2, 3), device=self.device)
        rotation_matrix[:, 0, 0] = torch.cos(rotation * np.pi / 180) * scale
        rotation_matrix[:, 0, 1] = -1 *torch.sin(rotation * np.pi / 180) * scale
        rotation_matrix[:, 1, 0] = torch.sin(rotation * np.pi / 180) * scale
        rotation_matrix[:, 1, 1] = torch.cos(rotation * np.pi / 180) * scale
        #other side
        rotation_t = (rotation - 180)
        rotation_matrix_t = torch.zeros((img.size()[0], 2, 3), device=self.device)
        rotation_matrix_t[:, 0, 0] = torch.cos(rotation_t * np.pi / 180) * scale
        rotation_matrix_t[:, 0, 1] = -1 *torch.sin(rotation_t * np.pi / 180) * scale
        rotation_matrix_t[:, 1, 0] = torch.sin(rotation_t * np.pi / 180) * scale
        rotation_matrix_t[:, 1, 1] = torch.cos(rotation_t * np.pi / 180) * scale
        print("ROTATION: ", rotation, rotation_t)

        return rotation, scale, rotation_matrix, rotation_matrix_t, cr

    def forward(self, img_o, template_o, plot=False):
        self.plot = plot
        img = img_o.clone() / 255
        template = template_o.clone() / 255

        if plot:
            #print(img.shape, template.shape)
            pass

        r, s, m, m_t, cr1 = self.get_rotation_scale(img, template)

        if plot:
            #print(m_t)
            print("R and S:", r, s)
            pass

        r_i = tu.warp(template, m, (img_o.shape[3], img_o.shape[2]))
        r_i_t = tu.warp(template, m_t, (img_o.shape[3], img_o.shape[2]))


        if plot:
            #print(img.shape, r_i.shape)
            #for b in range(img_o.size()[0]):
                #plt.title("Rota")
                #plt.imshow(r_i[b].transpose(0, 2).transpose(0, 1).detach().cpu().numpy(), cmap="gray")
                #plt.show()
            pass

        #CONV D
        pos, cr2 = self.calculate_shift(self.convD(img), self.convD(r_i))
        pos_t, cr2_t = self.calculate_shift(self.convD(img), self.convD(r_i_t))
        print("CR2s: ", cr2.argmax(), cr2_t.argmax(), torch.special.entr(cr2.abs()).mean(), torch.special.entr(cr2_t.abs()).mean())
        e = torch.special.entr(cr2.abs()).mean()
        e_t = torch.special.entr(cr2_t.abs()).mean()
        pos = pos if e < e_t else pos_t
        r_i = r_i if e < e_t else r_i_t

        if plot:
            #print(m_t)
            print("POS: ", pos)
            pass
        if plot:
            for b in range(img_o.size()[0]):
                print("printing batch: ", b)
                p_img = img_o[b].detach().transpose(0, 2).transpose(0, 1).cpu().numpy()
                p_temp = template_o[b].detach().transpose(0, 2).transpose(0, 1).cpu().numpy()

                x, y = pos[b, 1] * img_o.shape[3], pos[b, 0] * img_o.shape[2]

                tu.imshow("img", p_img / 255)
                tu.imshow("template", p_temp / 255)
                #imshow("rotated image torch", (torch.clamp(r_i[0].transpose(0, 2).transpose(0, 1).cpu().detach(), 0, 255).numpy()*0.5).astype(np.uint8) + p_img * 0.5)
                print("Final transform (r, s, x,y): ",r.item(), s.item(), x.item(), y.item())

                m2 = torch.tensor([[1, 0, -x], [0, 1, -y]], device=self.device)
                translated_image = tu.warp(r_i * 255, m2, (p_img.shape[1], p_img.shape[0]))

                im_copy2 = translated_image[b].transpose(0, 2).transpose(0, 1).detach().cpu().numpy() * 0.5 + p_img * 0.5

                tu.imshow("rotated translated template", (im_copy2 / 255))

        return(torch.concat([r.unsqueeze(1), s.unsqueeze(1), pos], 1), cr1, cr2)

