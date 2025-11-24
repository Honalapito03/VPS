#To do:
#1) Probabilistic coordinates
#2)Reference picture finding
#3) Rotation bug on l.27
#?)
#?) FMT, image taking

import torch
import cv2
import torch_utils as tu
import numpy as np
import math
import random
from torch_CC_model import Network
from torch_tranining import transform


class Coordinates():
    def __init__(self,x,y,s,r):
        self.x = x
        self.y = y
        self.s = s
        self.r = r
        self.phi = (r/180)*math.pi
        self.affine_matrix = None

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
        return Coordinates((self.x+other.x)/2, (self.y+other.y)/2,(self.s+other.s)/2,(averaged_angle/math.pi)*180) #make shorter
    
    def create_affine_matrix(self):
        self.affine_matrix = np.array(
            [
                [self.s*(math.cos(self.phi)), -self.s*(math.sin(self.phi)), self.x],
                [self.s*(math.sin(self.phi)),  self.s*(math.cos(self.phi)), self.y],
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
        self.px_mm = 1
        self.tiles = []
        self.pos = None
        self.last_ref = None
        self.current_image = 0
        self.device = "cpu"
        self.N = Network(argmax_beta=300.0, device=self.device).to(self.device)



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
            temp = FMT(pic, ref.image)
            pos_estimation.append(ref.coordinate.combine(temp)) #chain
            if ref.coverage_test(approx_pos) > last_cov:
                last_ref = ref
                last_cov = ref.coverage_test(approx_pos)

        pos = rec_merge(pos_estimation)

        if (last_ref.coverage_test(pos) < 0.5):
            tiles.append(map_tile(pic,pos))
