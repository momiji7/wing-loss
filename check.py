from tensorboardX import SummaryWriter
from datasets import GeneralDataset
import transforms
import torch
from resnet import *
from compute_nme import compute_nme
import cv2
from basic_args import obtain_args as obtain_basic_args
import datetime
from logger import Logger



def draw_points(img, pts, color=(255, 255, 0), radius=1, thickness=1, lineType=16, shift=4):
    
    draw_multiplier = 1<<shift
    for idx in pts:
        pt = [int(round(p*draw_multiplier)) for p in idx]  # for subpixel
        cv2.circle(img, tuple(pt), radius*draw_multiplier, color, 3, lineType, shift)
        cv2.circle(img, tuple(pt), radius*draw_multiplier, (255, 255, 0), thickness, lineType, shift)







def train(args):
  
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  
  
  print('Arguments : -------------------------------')
  for name, value in args._get_kwargs():
    print('{:16} : {:}'.format(name, value))
    
    
  # Data Augmentation    
  mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
  normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])


  # train_transform = [transforms.AugTransBbox(1, 0.5)]
  train_transform = [transforms.PreCrop(args.pre_crop_expand)]
  
  train_transform += [transforms.TrainScale2WH((args.crop_width, args.crop_height))]
  train_transform += [transforms.AugGaussianBlur(0.5, 17, 10)]

  train_transform += [transforms.ToTensor()]
  train_transform  = transforms.Compose( train_transform )


  # Training datasets
  train_data = GeneralDataset(args.num_pts, train_transform, args.train_lists)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

  
    
  net = resnet50(out_classes = args.num_pts*2)
  print("=> network :\n {}".format(net))
    
  print('arguments : {:}'.format(args))

  optimizer = torch.optim.SGD(net.parameters(), lr=args.LR, momentum=args.momentum,
                          weight_decay=args.decay, nesterov=args.nesterov)
    
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    
  net = net.cuda()
  net = torch.nn.DataParallel(net)
    
    
    
  print('--------------', len(train_loader))
  for epoch in range(3):
    for i , (inputs, target, meta) in enumerate(train_loader):
      
      nums_img = inputs.size()[0]
      for j in range(nums_img): 
        temp_img = inputs[j].permute(1,2,0)
        temp_img = temp_img.mul(255).numpy()
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)

        pts = []
        for d in range(args.num_pts):
          pts.append((target[j][0][2*d].item(), target[j][0][2*d+1].item()))
        bbox = [int(index[0].item())  for index in meta]
        draw_points(temp_img, pts, (0, 255, 255))
        cv2.rectangle(temp_img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),4)
        cv2.imwrite('{}-{}-{}.jpg'.format(epoch,i,j), temp_img)
      
      #if i > 5:
      #  break


  for a, v in enumerate(train_data.data_value):
   
    image = cv2.imread(v['image_path'])
    meta = v['meta']
    bbox = v['bbox']
    pts = []
    for d in range(args.num_pts):
      pts.append((meta.points[0, d], meta.points[1, d]))
    draw_points(image, pts, (0, 255, 255))
    cv2.rectangle(image,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,255,0),4)
    cv2.imwrite('ori_{}.jpg'.format(a), image)
    
    #if a > 15:
    #    break
    
  


if __name__ == '__main__':
   args = obtain_basic_args()
   train(args)