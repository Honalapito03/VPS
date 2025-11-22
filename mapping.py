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

resolution = 100

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
        print(vector,self.affine_matrix)
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
    def __init__(self,image:np.ndarray,coordinate:Coordinates):
        self.image = image
        self.coordinate = coordinate
        self.coordinate.create_affine_matrix()
        self.r = resolution / 2 * coordinate.s

    def coverage_test(self, coordinate:Coordinates): 
        r1 = resolution / 2 * coordinate.s

        required_distance = (self.r + r1) 
        x_y_distance = ((self.coordinate.x - coordinate.x)**2 + (self.coordinate.y - coordinate.y)**2 )**0.5

        return (required_distance - x_y_distance)/required_distance



        
def image_taking(image_count):
    return image_count

def ref_pic_find(coordinates:Coordinates, tiles:list[map_tile]) -> list[map_tile]:
    good_tiles = []
    for tile in tiles:
        if tile.coverage_test(coordinates) > 1:
            good_tiles.append(tile)
    return good_tiles
#Can be optimized further

def FMT(picture:np.ndarray, template:np.ndarray):
    r_x = random.randint(-50,50)
    r_y = random.randint(-50,50)

    print("FMT ",template)

    return Coordinates(110,0,1,30)

def rec_merge(coordinates:list):
    if len(coordinates) > 2:
        f_ck_recursion = rec_merge(coordinates[:len(coordinates)//2])
        i_hate_recursion = rec_merge(coordinates[len(coordinates)//2:])
        return f_ck_recursion.merge(i_hate_recursion)
    elif len(coordinates) == 2:
        return coordinates[0].merge(coordinates[1])
    else:
        return coordinates[0]

def main():

    image_count = 0

    tiles=[map_tile(image_taking(image_count),Coordinates(0,0,1,0))]

    pos = Coordinates(0,0,0,0)

    px_mm = 1

    last_ref = tiles[0]
    for e in range(12):

        image_count += 1

        pic = image_taking(image_count)
        approx_pos = last_ref.coordinate.combine(FMT(pic, last_ref.image))
        refs = ref_pic_find(approx_pos,tiles)
        last_cov = last_ref.coverage_test(approx_pos)
        pos_estimation = [approx_pos]
    
        for ref in refs:
            if ref != last_ref:
                temp = FMT(pic, ref.image)
                pos_estimation.append(ref.coordinate.combine(temp)) #chain
                if ref.coverage_test(approx_pos) > last_cov:
                    last_ref = ref
                    last_cov = ref.coverage_test(approx_pos)

        pos = rec_merge(pos_estimation)

        if (last_ref.coverage_test(pos) < 0.5):
            tiles.append(map_tile(pic,pos))
            last_ref = tiles[-1]

        print(pos.x,pos.y,pos.s,pos.r)
        print(len(tiles))
        print(last_ref.image)
        print()

main()

            
