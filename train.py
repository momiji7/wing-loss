from tensorboardX import SummaryWriter
from datasets import GeneralDataset
import transforms
import torch
import torch.nn as nn
from resnet import *
from resnet import FrozenBatchNorm2d, BatchNorm2d_para
from compute_nme import compute_nme
from basic_args import obtain_args as obtain_basic_args
import datetime
from logger import Logger
from meter import AverageMeter
from wingloss import wing_loss
from copy import deepcopy


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

  train_transform = [transforms.AugTransBbox(args.transbbox_prob, args.transbbox_percent)]
  train_transform += [transforms.PreCrop(args.pre_crop_expand)]
  train_transform += [transforms.TrainScale2WH((args.crop_width, args.crop_height))]
  train_transform += [transforms.AugHorizontalFlip(args.flip_prob)]
  if args.rotate_max:
    train_transform += [transforms.AugRotate(args.rotate_max)]
  train_transform += [transforms.AugGaussianBlur(args.gaussianblur_prob, args.gaussianblur_kernel_size, args.gaussianblur_sigma)]
  train_transform += [transforms.ToTensor(), normalize]
  train_transform  = transforms.Compose( train_transform )


  eval_transform  = transforms.Compose([transforms.PreCrop(args.pre_crop_expand), transforms.TrainScale2WH((args.crop_width, args.crop_height)),  transforms.ToTensor(), normalize])

  # Training datasets
  train_data = GeneralDataset(args.num_pts, train_transform, args.train_lists)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

  # Evaluation Dataloader
  eval_loaders = []

  for eval_ilist in args.eval_lists:
    eval_idata = GeneralDataset(args.num_pts, eval_transform, eval_ilist)
    eval_iloader = torch.utils.data.DataLoader(eval_idata, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    eval_loaders.append(eval_iloader)
       
    
  # net = resnet50(out_classes = args.num_pts*2, pretrained=True, norm_layer = FrozenBatchNorm2d)
  net = resnet50(out_classes = args.num_pts*2)
 
  #ct = 0
  #for child in net.children():
  #  if ct < 3:
  #    ct += 1
  #    for param in child.parameters():
  #      param.requires_grad = False  
       
    
  logger.log("=> network :\n {}".format(net))
    
  logger.log('arguments : {:}'.format(args))

  optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.LR, momentum=args.momentum,
                          weight_decay=args.decay, nesterov=args.nesterov)
    
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
   
  criterion = wing_loss(args)  
  # criterion = torch.nn.MSELoss(reduce=True)
    
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
    
  for epoch in range(start_epoch, args.epochs):
    scheduler.step()
    
    net.train()
    
    # train
    img_prediction = []
    img_target = []
    train_losses = AverageMeter()
    for i , (inputs, target) in enumerate(train_loader):
      
      target = target.squeeze(1)
      inputs = inputs.cuda()
      target = target.cuda()
      #print(inputs.size())
      #ssert 1==0
    
      prediction = net(inputs)
            
      loss = criterion(prediction, target) 
      train_losses.update(loss.item(), inputs.size(0))
      
      prediction = prediction.detach().to(torch.device('cpu')).numpy()
      target = target.detach().to(torch.device('cpu')).numpy()
        
      for idx in range(inputs.size()[0]):        
        img_prediction.append(prediction[idx,:])
        img_target.append(target[idx,:])
     
        
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        
      if i % args.print_freq == 0 or i+1 == len(train_loader):
        logger.log('[train Info]: [epoch-{}-{}][{:04d}/{:04d}][Loss:{:.2f}]'.format(epoch, args.epochs, i, len(train_loader), loss.item()))
         
    train_nme = compute_nme(args.num_pts, img_prediction, img_target)
    logger.log('epoch {:02d} completed!'.format(epoch))
    logger.log('[train Info]: [epoch-{}-{}][Avg Loss:{:.6f}][NME:{:.2f}]'.format(epoch, args.epochs, train_losses.avg, train_nme*100))
    tfboard_writer.add_scalar('Average Loss', train_losses.avg, epoch)
    tfboard_writer.add_scalar('NME', train_nme*100, epoch) # traing data nme
     
    # save checkpoint           
    filename = 'epoch-{}-{}.pth'.format(epoch, args.epochs)
    save_path = logger.path('model') / filename
    torch.save({
      'epoch': epoch,
      'args' : deepcopy(args),
      'state_dict': net.state_dict(),
      'scheduler' : scheduler.state_dict(),
      'optimizer' : optimizer.state_dict(),
    }, logger.path('model') / filename)  
    logger.log('save checkpoint into {}'.format(filename))
    last_info = torch.save({
      'epoch': epoch,
      'last_checkpoint': save_path
    }, logger.last_info())
                
    # eval           
    logger.log('Basic-Eval-All evaluates {} dataset'.format(len(eval_loaders)))
    
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
        
          for idx in range(inputs.size()[0]):        
            eval_prediction.append(prediction[idx,:])
            eval_target.append(target[idx,:])
          if i_batch % args.print_freq == 0 or i+1 == len(loader):
            logger.log('[Eval Info]: [epoch-{}-{}][{:04d}/{:04d}][Loss:{:.2f}]'.format(epoch, args.epochs, i, len(loader), loss.item()))
            
      eval_nme = compute_nme(args.num_pts, eval_prediction, eval_target)
      logger.log('[Eval Info]: [evaluate the {}/{}-th dataset][epoch-{}-{}][Avg Loss:{:.6f}][NME:{:.2f}]'.format(i, len(eval_loaders) ,epoch, args.epochs, eval_losses.avg, eval_nme*100)) 
      tfboard_writer.add_scalar('eval_nme/{}'.format(i), eval_nme*100, epoch)
    
  logger.close()
    

if __name__ == '__main__':
   args = obtain_basic_args()
   train(args)
  