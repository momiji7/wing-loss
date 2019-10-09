import torch.utils.data as data
import json
import os
import cv2
import torch
import numpy as np
from point_meta import Point_Meta
from util import *

class GeneralDataset(data.Dataset):
    
    def __init__(self, num_pts, transform, json_path):
              
        if isinstance(json_path, str):
            json_path = [json_path]
        
        self.data = {}
        for file_path in json_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.data.update(data)
        
        self.data_value = list(self.data.values())
        self.data_len = len(self.data_value)
        self.num_pts = num_pts
        self.transform = transform
        
        # convert list to point_meta
        for v in self.data_value:
            landmarks = v['landmarks']
            
            
            
            # assert 2*num_pts == len(landmarks), 'The lenght of landmarks {} is not {}'.format(len(landmarks), 2*num_pts)
            target = np.zeros((3, num_pts), dtype='float32')
            for idx in range(num_pts):
                target[0, idx] = float(landmarks[2*idx])
                target[1, idx] = float(landmarks[2*idx+1])
                target[2, idx] = 1
            
            box = box2square(v['bbox'])
            # box = v['bbox']
            meta = Point_Meta(num_pts, target, box)
            v['meta'] = meta
    
    def __len__(self):
        return self.data_len
    
        
    def __getitem__(self, index):
        
        image = cv2.imread(self.data_value[index]['image_path'])
        target = self.data_value[index]['meta']
        
        #print(self.data_value[index]['image_path'])
        #print(image.shape)
        if self.transform is not None:
            image, target = self.transform(image, target)
            
        crop_save_wh = target.crop_save_wh
        cropped_size = torch.IntTensor( [crop_save_wh[1], crop_save_wh[0], crop_save_wh[2], crop_save_wh[3]]) # H, W, Cropped_[x1,y1]
        
        if target.points is not None:
            points = target.points.copy()
            points = torch.FloatTensor(points)
            points_t = points.t()
            pts = points_t[:,:2]
            mask = points_t[:,2].unsqueeze(1)
            pts_masked = pts*mask
            
        torch_index = torch.IntTensor([index])
        mask = mask.type(torch.ByteTensor)
        # print(image.size())    
        return image, pts_masked.reshape(1, -1), # mask, cropped_size  # x1, y1, x2, y2 ... 
        
        
        
            