from tensorboardX import SummaryWriter
from datasets import GeneralDataset
import transforms
import torch
from resnet import *
from wingloss import wing
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
  
  tfboard_writer = SummaryWriter()
  logname = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'))
  logger = Logger(args.save_path, logname)
  logger.log('Arguments : -------------------------------')
  for name, value in args._get_kwargs():
    logger.log('{:16} : {:}'.format(name, value))
    
    
  # Data Augmentation    
  mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
  normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

  train_transform  = [transforms.PreCrop(args.pre_crop_expand)]
  train_transform += [transforms.TrainScale2WH((args.crop_width, args.crop_height))]
  train_transform += [transforms.AugScale(args.scale_prob, args.scale_min, args.scale_max)]
  #if args.arg_flip:
  #  train_transform += [transforms.AugHorizontalFlip()]
  if args.rotate_max:
    train_transform += [transforms.AugRotate(args.rotate_max)]
  train_transform += [transforms.AugCrop(args.crop_width, args.crop_height, args.crop_perturb_max, mean_fill)]
  train_transform += [transforms.ToTensor()]
  train_transform  = transforms.Compose( train_transform )

  eval_transform  = transforms.Compose([transforms.PreCrop(args.pre_crop_expand), transforms.TrainScale2WH((args.crop_width, args.crop_height)),  transforms.ToTensor(), normalize])

  # Training datasets
  train_data = GeneralDataset(args.num_pts, train_transform, args.train_lists)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

  
    
  net = resnet50(out_classes = args.num_pts*2)
  logger.log("=> network :\n {}".format(net))
    
  logger.log('arguments : {:}'.format(args))

  optimizer = torch.optim.SGD(net.parameters(), lr=args.LR, momentum=args.momentum,
                          weight_decay=args.decay, nesterov=args.nesterov)
    
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    
  net = net.cuda()
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
    
  print('--------------', len(train_loader))
  for epoch in range(3):
    for i , (inputs, target) in enumerate(train_loader):

      nums_img = inputs.size()[0]
      print(target.size())
      for j in range(nums_img): 
        temp_img = inputs[j].permute(1,2,0)
        temp_img = temp_img.mul(255).numpy()
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)

        pts = []
        for d in range(81):
          pts.append((target[j][0][2*d].item(), target[j][0][2*d+1].item()))
        print('0000000000000000000')
        print(pts)
        draw_points(temp_img, pts, (0, 255, 255))
        cv2.imwrite('{}-{}.jpg'.format(epoch,j), temp_img)


  for a, v in enumerate(train_data.data_value):
   
    image = cv2.imread(v['image_path'])
    meta = v['meta']
    pts = []
    for d in range(81):
      pts.append((meta.points[0, d], meta.points[1, d]))
    draw_points(image, pts, (0, 255, 255))
    cv2.imwrite('ori_{}.jpg'.format(a), image)
     
    
  


if __name__ == '__main__':
   args = obtain_basic_args()
   train(args)