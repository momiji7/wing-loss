

import torch



def wing(args, prediction, gt):
    diff = torch.abs(prediction - gt)
    if diff < args.wingloss_w:
        loss = args.wingloss_w * torch.log(1 + diff/args.wingloss_epsilon)
    else:
        loss = diff - args.wingloss_const
        
    return torch.sum(loss)
    