import torch
import torch.nn as nn
import math


class wing_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w = args.wingloss_w
        self.epsilon = args.wingloss_epsilon
        self.constant = args.wingloss_w - args.wingloss_w * math.log(1 + args.wingloss_w / args.wingloss_epsilon)
        # print(self.w, self.epsilon, self.constant)
        
    def forward(self, prediction, gt):
        
        #prediction_mask = torch.masked_select(prediction , mask)
        #gt_mask = torch.masked_select(gt , mask)
        
        #print(prediction.size())
        #print(mask.size())
        #print(gt.size())
        #assert 1==0
        
        diff = torch.abs(prediction - gt)
        
        loss = torch.where(diff < self.w, self.w * torch.log(1 + diff/self.epsilon), diff - self.constant)
        
        loss = loss.view(-1)
        
        #diff = diff.view(-1)
        #loss = torch.norm(diff)
        
               
        return torch.mean(loss)
