#To do:
#1) Probabilistic coordinates
#2) Rotation precision error handling
#3) Scale diff max limit

import torch
import cv2
import torch_utils as tu
import numpy as np
import math
import random
import xlsxwriter

from torch_CC_model import Network
from torch_tranining import transform


class Coordinates():
    def __init__(self,x,y,s,r, rs_snr=1, xy_snr=1):
        self.x = x
        self.y = y
        self.s = s
        self.r = r
        self.phi = (r/180)*math.pi
        self.affine_matrix = None

        self.rs_snr = rs_snr
        self.xy_snr = xy_snr

    def combine(self, other): #other is Coordinates :)
        vector = np.array([other.x,other.y,1])
        t_vector =  self.affine_matrix @ vector
        return Coordinates(t_vector[0],t_vector[1],self.s*other.s,(self.r+other.r)%360)
    
    def merge(self,other):
        sin_r = math.sin(self.phi)
        cos_r = math.cos(self.phi)

        sin_other_r = math.sin(other.phi) 
        cos_other_r = math.cos(other.phi)

        average_sin = (sin_r + sin_other_r) / 2
        average_cos = (cos_r + cos_other_r) / 2

        averaged_angle = math.atan2(average_sin, average_cos)
        return Coordinates( (self.x+other.x)/2, 
                            (self.y+other.y)/2,
                            (self.s+other.s)/2,
                            (averaged_angle/math.pi)*180, 
                            (self.rs_snr + other.rs_snr)/2,
                            (self.xy_snr + other.xy_snr)/2
                           ) #make shorter
    

    def create_affine_matrix(self):
        self.affine_matrix = np.array(
            [
                [(math.cos(self.phi)) / self.s, - (math.sin(self.phi)) / self.s, self.x],
                [(math.sin(self.phi)) / self.s,  (math.cos(self.phi)) / self.s, self.y],
                [                        0,                         0,           1],
            ]
        )

class map_tile():
    def __init__(self,image:np.ndarray,coordinate:Coordinates, resolution):
        self.image = image
        self.coordinate = coordinate
        self.coordinate.create_affine_matrix()
        self.r = resolution / 2 * coordinate.s

    def coverage_test(self, coordinate:Coordinates, resolution): 
        r1 = resolution / 2 * coordinate.s

        required_distance = (self.r + r1) 
        x_y_distance = ((self.coordinate.x - coordinate.x)**2 + (self.coordinate.y - coordinate.y)**2 )**0.5

        return (required_distance - x_y_distance)/required_distance


class Mapper():
    def __init__(self, ):
        self.resolution = 100
        self.x_res = 1
        self.y_res = 1
        self.px_mm = 1
        self.tiles = []
        self.pos = None
        self.last_ref = None
        self.current_image = 0
        self.device = "cpu"
        self.N = Network(argmax_beta=1000.0, device=self.device).to(self.device)
        self.take_history = False

        self.h_x = []
        self.h_y = []
        self.h_s = []
        self.h_r = []
        self.h_last_ref = []
        self.h_last_ref_cov = []
        self.h_ref_count = []
        self.h_ref_avr_cov = []

        self.gt_x = []
        self.gt_y = []
        self.gt_s = []
        self.gt_r = []



    def start(self):
        #Initial map tile
        #main: outside of for
        self.tiles=[map_tile(self.current_image, Coordinates(0,0,1,0), self.resolution)]
        self.pos = Coordinates(0,0,0,0)
        self.last_ref = self.tiles[0]


    def loop_step(self):
        pic = self.current_image
        
        #early testing
        self.current_image += 1

        approx_pos = self.last_ref.coordinate.combine(self.FMT(self.last_ref.image, pic))
        refs = self.ref_pic_find(approx_pos, self.tiles)
        last_cov = self.last_ref.coverage_test(approx_pos, self.resolution)
        pos_estimation = [approx_pos]
    
        for ref in refs:
            if ref != self.last_ref:
                temp = self.FMT(ref.image, pic)
                pos_estimation.append(ref.coordinate.combine(temp)) #chain
                if ref.coverage_test(approx_pos, self.resolution) > last_cov:
                    self.last_ref = ref
                    last_cov = ref.coverage_test(approx_pos, self.resolution)

        self.pos = self.rec_merge(pos_estimation)

        if (self.last_ref.coverage_test(self.pos, self.resolution) < 0.85):
            self.tiles.append(map_tile(pic, self.pos, self.resolution))
            self.last_ref = self.tiles[-1]

        print("Calculated: ", self.pos.x,self.pos.y,self.pos.s,self.pos.r)
        print("data: ", len(self.tiles), len(refs), self.last_ref.coverage_test(self.pos, self.resolution))
        
        print("----")

        #History
        if self.take_history:
            self.h_x.append(self.pos.x * self.px_mm)
            self.h_y.append(self.pos.y * self.px_mm)
            self.h_s.append(self.pos.s)
            self.h_r.append(self.pos.r)
            self.h_last_ref.append(self.tiles.index(self.last_ref))
            self.h_last_ref_cov.append(self.last_ref.coverage_test(self.pos, self.resolution))
            self.h_ref_count.append(len(refs))
            covs = [tile.coverage_test(self.pos, self.resolution) for tile in refs] 
            if len(covs) > 0:
                self.h_ref_avr_cov.append(sum(covs)/len(covs))
            else:
                self.h_ref_avr_cov.append(0)



    def ref_pic_find(self, coordinates:Coordinates, tiles:list[map_tile]) -> list[map_tile]:
        good_tiles = []
        for tile in tiles:
            if tile.coverage_test(coordinates, self.resolution) > 0.7:
                good_tiles.append(tile)
        return good_tiles
    #Can be optimized further


    def FMT(self, picture:np.ndarray, template:np.ndarray) -> Coordinates:
        print("template: ", picture)
        img_o = torch.tensor(picture, dtype=torch.float32, device=self.device).unsqueeze(2).transpose(0, 2).transpose(1, 2).unsqueeze(0)
        temp_o = torch.tensor(template, dtype=torch.float32, device=self.device).unsqueeze(2).transpose(0, 2).transpose(1, 2).unsqueeze(0)
        res, cr1, cr2 = self.N(img_o, temp_o, False)
        print("FMT result: ", res, "signal to noise: ", tu.signal_to_noise(cr1), tu.signal_to_noise(cr2))

        return Coordinates(res[0][2].item() * self.x_res, res[0][3].item() * self.y_res, res[0][1].item(), res[0][0].item())


    def rec_merge(self, coordinates:list) -> Coordinates:
        if len(coordinates) > 2:
            f_ck_recursion = self.rec_merge(coordinates[:len(coordinates)//2])
            i_hate_recursion = self.rec_merge(coordinates[len(coordinates)//2:])
            return f_ck_recursion.merge(i_hate_recursion)
        elif len(coordinates) == 2:
            return coordinates[0].merge(coordinates[1])
        else:
            return coordinates[0]
        
    def add_gt(self, x, y, s, r):
        self.gt_x.append(x)
        self.gt_y.append(y)
        self.gt_s.append(s)
        self.gt_r.append(r)
    
    def export_history(self, filename):
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet()

        headers = ['est_x_mm', 'est_y_mm', 'est_s', 'est_r', 'last_ref_index', 'last_ref_cov', 'ref_count', 'ref_avr_cov', 'gt_x_mm', 'gt_y_mm',  'gt_s', 'gt_r']
        for col_num, header in enumerate(headers):
            worksheet.write(0, col_num, header)

        for row in range(len(self.h_x)):
            worksheet.write(row + 1, 0, self.h_x[row])
            worksheet.write(row + 1, 1, self.h_y[row])
            worksheet.write(row + 1, 2, self.h_s[row])
            worksheet.write(row + 1, 3, self.h_r[row])
            worksheet.write(row + 1, 4, self.h_last_ref[row])
            worksheet.write(row + 1, 5, self.h_last_ref_cov[row])
            worksheet.write(row + 1, 6, self.h_ref_count[row])
            worksheet.write(row + 1, 7, self.h_ref_avr_cov[row])
            if row < len(self.gt_x):
                worksheet.write(row + 1, 8, self.gt_x[row])
                worksheet.write(row + 1, 9, self.gt_y[row])
                worksheet.write(row + 1, 10, self.gt_s[row])
                worksheet.write(row + 1, 11, self.gt_r[row])

        workbook.close()


if __name__ == "__main__":
    m = Mapper()
    m.start()
    for _ in range(12):
        m.loop_step()

            
