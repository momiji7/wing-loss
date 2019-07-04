

import os, sys, time, random, argparse

def obtain_args():
  parser = argparse.ArgumentParser(description='Wing loss', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--train_lists',      type=str,   nargs='+',      help='The list file path to the video training dataset.')
  parser.add_argument('--eval_lists',       type=str,   nargs='+',      help='The list file path to the image testing dataset.')
  parser.add_argument('--num_pts',          type=int,                   help='Number of point.')

  # Data Transform
  parser.add_argument('--pre_crop_expand',  type=float, default=0.2,    help='parameters for pre-crop expand ratio')
  parser.add_argument('--scale_prob',       type=float, default=0.2,    help='argument scale probability.')
  parser.add_argument('--scale_min',        type=float, default=0.9,    help='argument scale : minimum scale factor.')
  parser.add_argument('--scale_max',        type=float, default=1.1,    help='argument scale : maximum scale factor.')
  parser.add_argument('--rotate_max',       type=int,   default=30,     help='argument rotate : maximum rotate degree.')
  parser.add_argument('--crop_height',      type=int,   default=256,    help='argument crop : crop height.')
  parser.add_argument('--crop_width',       type=int,   default=256,    help='argument crop : crop width.')
  parser.add_argument('--crop_perturb_max', type=int,   default=10,     help='argument crop : center of maximum perturb distance.')
  parser.add_argument('--flip_prob',        type=float, default=0.5,    help='argument flip probability.')
  parser.add_argument('--gaussianblur_prob',type=float, default=0.5,    help='argument gaussianblur probability.')  
  parser.add_argument('--gaussianblur_kernel_size', type=int,default=5, help='argument gaussianblur kernel_size.') 
  parser.add_argument('--gaussianblur_sigma',type=float,  default=1,      help='argument gaussianblur sigma.')
  parser.add_argument('--transbbox_prob',   type=float, default=1,      help='argument transbbox probability.')
  parser.add_argument('--transbbox_percent',type=float, default=0.025,  help='argument transbbox percent.')
    
    
  # Optimization options
  parser.add_argument('--eval_once',        action='store_true',        help='evaluation only once for evaluation ')
  parser.add_argument('--error_bar',        type=float,                 help='For drawing the image with large distance error.')
  parser.add_argument('--batch_size',       type=int,   default=8,      help='Batch size for training.')
  # Checkpoints
  parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
  parser.add_argument('--save_path',        type=str,                   help='Folder to save checkpoints and log.')
  # Acceleration
  parser.add_argument('--workers',          type=int,   default=8,      help='number of data loading workers (default: 2)')
  
  # Optimizer
  parser.add_argument('--LR',               type=float, default=0.00003,help='Learning rate for optimizer.')
  parser.add_argument('--momentum',         type=float, default=0.9,    help='Momentum for optimizer.')
  parser.add_argument('--decay',            type=float, default=0.0005, help='Decay for optimizer.')
  parser.add_argument('--nesterov',         action='store_true',        help='Using nesterov for optimizer.')
  parser.add_argument('--epochs',           type=int,   default=50,     help='Epochs for training')
  # lr_scheduler
  parser.add_argument('--schedule',         type=int,   nargs='+',      
                     default=[30, 40],                                  help='The list file path to the video training dataset.')
  parser.add_argument('--gamma',            type=float, default=0.1,    help='Decay for learning rate.')
  # loss
  parser.add_argument('--wingloss_w',       type=float, default=10,     help='W parameter for optimizer.')
  parser.add_argument('--wingloss_epsilon', type=float, default=2,      help='Epsilon parameter.')  
    
    
  # log file     
  args = parser.parse_args()
 

  #state = {k: v for k, v in args._get_kwargs()}
  #Arguments = namedtuple('Arguments', ' '.join(state.keys()))
  #arguments = Arguments(**state)
  return args
