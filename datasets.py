import torch.utils.data as data
import json
import os
import cv2
import torch
import numpy as np

class GeneralDataset(data.Dataset):
    
    def __init__(self, num_pts, transform, json_path):
        assert os.path.isfile(json_path), 'Can not find {}'.format(json_path)
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.data_value = list(self.data.values())
        self.data_len = len(data_value)
        self.num_pts = num_pts
        
        # convert list to point_meta
        for v in self.data_value:
            landmarks = v['landmarks']
            
            assert 2*num_pts == len(landmarks), 'The lenght of landmarks {} is not {}'.format(len(landmarks), 2*num_pts)
            np.zeros((3, n_points), dtype='float32')
            for idx in range(num_pts):
                np[0, idx] = float(landmarks[2*idx])
                np[1, idx] = float(landmarks[2*idx+1])
                np[2, idx] = 1
            
            meta = Point_Meta(num_pts, landmarks, v['bbox'])
            v['meta'] = meta
    
    def __len__(self):
        return self.data_len
    
        
    def __getitem__(self, index):
        
        image = cv2.imread(self.data_value[index]['image_path'])
        target = self.data_value['meta']
        
        if self.transform is not None:
            image, target = self.transform(image, target)
            
            
        return image, target
        
        
        
            