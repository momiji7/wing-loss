from tensorboardX import SummaryWriter
from datasets import GeneralDataset
import transforms
import torch
from resnet import *
from compute_nme import compute_nme
from basic_args import obtain_args as obtain_basic_args
import datetime
from logger import Logger
from meter import AverageMeter
from wingloss import wing_loss
from copy import deepcopy
from cnn6 import Net
import cv2


def draw_points(img, pts, color=(255, 255, 0), radius=1, thickness=1, lineType=16, shift=4):
    
  draw_multiplier = 1<<shift
  for idx in pts:
    pt = [int(round(p*draw_multiplier)) for p in idx]  # for subpixel
    cv2.circle(img, tuple(pt), radius*draw_multiplier, color, 3, lineType, shift)
    cv2.circle(img, tuple(pt), radius*draw_multiplier, (255, 255, 0), thickness, lineType, shift)



def test(args):
  
  logname = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'))
  logger = Logger(args.save_path, logname)
  # Data Augmentation    
  mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
  normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

  eval_transform  = transforms.Compose([transforms.PreCrop(args.pre_crop_expand), transforms.TrainScale2WH((args.crop_width, args.crop_height)),  transforms.ToTensor()])

 
  eval_loaders = []

  for eval_ilist in args.eval_lists:
    eval_idata = GeneralDataset(args.num_pts, eval_transform, eval_ilist)
    eval_iloader = torch.utils.data.DataLoader(eval_idata, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    eval_loaders.append(eval_iloader)
    
    
  net = Net(args)

  optimizer = torch.optim.SGD(net.parameters(), lr=args.LR, momentum=args.momentum,
                          weight_decay=args.decay, nesterov=args.nesterov)
    
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
   
  criterion = wing_loss(args)  
  criterion = torch.nn.MSELoss(reduce=True)

    
  net = net.cuda()
  criterion = criterion.cuda()
  net = torch.nn.DataParallel(net)
    
    
  last_info = logger.last_info()
  if last_info.exists():
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info = torch.load(last_info)
    start_epoch = last_info['epoch'] + 1
    checkpoint  = torch.load(last_info['last_checkpoint'])
    assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info, checkpoint['epoch'])
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    logger.log("=> load-ok checkpoint '{:}' (epoch {:}) done" .format(logger.last_info(), checkpoint['epoch']))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch = 0 

 
  for i, loader in enumerate(eval_loaders):
                
    eval_losses = AverageMeter()
    eval_prediction = []
    eval_target = []    
    with torch.no_grad():
      net.eval()
      for i_batch , (inputs, target) in enumerate(loader):
            
        target = target.squeeze(1)
        inputs = inputs.cuda()
        target = target.cuda()
        prediction = net(inputs)
        loss = criterion(prediction, target) 
        eval_losses.update(loss.item(), inputs.size(0))
        
        prediction = prediction.detach().to(torch.device('cpu')).numpy()
        target = target.detach().to(torch.device('cpu')).numpy()
        inputs = inputs.cpu()
        
        for idx in range(inputs.size()[0]): 
            
          temp_img = inputs[idx].permute(1,2,0)
          temp_img = temp_img.mul(255).numpy()
          temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR) 
          temp_img = cv2.resize(temp_img, (256, 256))
          pts = []
          pts_pre = []
          #print(prediction[idx])
          #print(target[idx])
          for d in range(args.num_pts):
            pts.append((target[idx][2*d]*4, target[idx][2*d+1]*4))
            pts_pre.append((prediction[idx][2*d]*4, prediction[idx][2*d+1]*4))
                                
          draw_points(temp_img, [(target[idx][2*36]*4, target[idx][2*36+1]*4)], (0, 255, 255))
          draw_points(temp_img, [(target[idx][2*45]*4, target[idx][2*45+1]*4)], (0, 255, 255))
          draw_points(temp_img, pts_pre, (255, 0, 255))
          cv2.imwrite('{}-{}.jpg'.format(0,idx), temp_img)
            
          eval_prediction.append(prediction[idx,:])
          eval_target.append(target[idx,:])
            
            
        assert 1==0
                   
      eval_nme = compute_nme(args.num_pts, eval_prediction, eval_target)
      print(eval_nme*100)
    
    
  logger.close()
    

if __name__ == '__main__':
   args = obtain_basic_args()
   test(args)
  