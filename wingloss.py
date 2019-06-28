import torch
import torch.nn as nn
import math


class wing_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w = args.wingloss_w
        self.epsilon = args.wingloss_epsilon
        self.constant = args.wingloss_w - args.wingloss_w * math.log(1 + args.wingloss_w / args.wingloss_epsilon)
        
    def forward(self, prediction, gt):
        diff = torch.abs(prediction - gt)
        loss = diff
        idx =  diff < self.w
        
        loss[idx] = self.w * torch.log(1 + diff[idx]/self.epsilon)
        idx = (idx + 1)%2
        loss[idx] = diff[idx] - self.constant
        print(loss)
        return torch.sum(loss)
