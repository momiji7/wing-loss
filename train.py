
from tensorboardX import SummaryWriter
from tensorboardX import SummaryWriter
import transforms

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
  mean_fill   = tuple( [int(x*255) for x in [0.406, 0.456, 0.485] ] )
  normalize   = transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                      std=[0.225, 0.224, 0.229])

  train_transform  = [transforms.PreCrop(args.pre_crop_expand)]
  train_transform += [transforms.TrainScale2WH((args.crop_width, args.crop_height))]
  train_transform += [transforms.AugScale(args.scale_prob, args.scale_min, args.scale_max)]
  #if args.arg_flip:
  #  train_transform += [transforms.AugHorizontalFlip()]
  if args.rotate_max:
    train_transform += [transforms.AugRotate(args.rotate_max)]
  train_transform += [transforms.AugCrop(args.crop_width, args.crop_height, args.crop_perturb_max, mean_fill)]
  train_transform += [transforms.ToTensor(), normalize]
  train_transform  = transforms.Compose( train_transform )