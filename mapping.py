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
from torch_CC_model import Network
from torch_tranining import transform

resolution = 1000

class Coordinates():
    def __init__(self,x,y,s,r):
        self.x = x
        self.y = y
        self.s = s
        self.r = r

    def combine(self, other): #other is Coordinates :)
        return Coordinates[self.x+other.x, self.y+other.y,self.s+other.s,(self.r+other.r)%360]
    
    def merge(self,other):
        return Coordinates[(self.x+other.x)/2, (self.y+other.y)/2,(self.s+other.s)/2,(self.r+other.r)%360] #bug :) Rotation averaging?

class map_tile():
    def __init__(self,image:np.ndarray,coordinate:Coordinates):
        self.image = image
        self.coordinate = coordinate
        self.r = resolution / 2 * coordinate.s

    def coverage_test(self, coordinate:Coordinates): 
        r1 = resolution / 2 * coordinate.s

        required_distance = (self.r + r1) / 2
        x_y_distance = ((self.coordinate.x - coordinate.x)**2 + (self.coordinate.y - coordinate.y)**2 )**0.5

        return((x_y_distance / required_distance) < 0.5)



        
def image_taking():
    pass

def ref_pic_find(coordinates:Coordinates, tiles:list[map_tile]) -> list[map_tile]:
    pass

def FMT(picture:np.ndarray, template:np.ndarray):
    pass

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
    tiles=[map_tile(image_taking(),Coordinates(0,0,0,0))]

    pos = Coordinates(0,0,0,0)

    px_mm = 1

    while True:
        pic = image_taking()
        refs = ref_pic_find(pos,tiles)

        pos_estimation = []

        for ref in refs:
            temp = FMT(pic, ref.image)
            pos_estimation.append(ref.coordinate.combine(temp)) #chain
        
        pos = rec_merge(pos_estimation)

        if (not any([ref.coverage_test(pos) for ref in refs])):
            tiles.append(map_tile(pic,pos))

            
