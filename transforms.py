import torch
import sys, math
import numpy as np
import numbers
import cv2
import random


class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img, points):
    for t in self.transforms:
      img, points = t(img, points)
    return img, points


class Normalize(object):
  """Normalize an tensor image with mean and standard deviation.
  Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  Args:
    mean (sequence): Sequence of means for (R, G, B) channels respecitvely.
    std (sequence): Sequence of standard deviations for (R, G, B) channels
      respecitvely.
  """

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensors, points):
    """
    Args:
      tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
      Tensor: Normalized image.
    """
    # TODO: make efficient
    if isinstance(tensors, list): is_list = True
    else:                         is_list, tensors = False, [tensors]

    for tensor in tensors:
      for t, m, s in zip(tensor, self.mean, self.std):
        t.sub_(m).div_(s)
    
    if is_list == False: tensors = tensors[0]

    return tensors, points



class PreCrop(object):

  def __init__(self, expand_ratio):
    assert expand_ratio is None or isinstance(expand_ratio, numbers.Number), 'The expand_ratio should not be {}'.format(expand_ratio)
    if expand_ratio is None:
      self.expand_ratio = 0
    else:
      self.expand_ratio = expand_ratio
    assert self.expand_ratio >= 0, 'The expand_ratio should not be {}'.format(expand_ratio)

  def __call__(self, img, point_meta):
    ## AugCrop has something wrong... For unsupervised data

 
    point_meta = point_meta.copy()
    #print(img.shape)
    h, w, _ = img.shape  # h*w*c
    box = point_meta.get_box()
    #print(box)
    face_ex_w, face_ex_h = (box[2] - box[0]) * self.expand_ratio, (box[3] - box[1]) * self.expand_ratio
    x1, y1 = int(max(math.floor(box[0]-face_ex_w), 0)), int(max(math.floor(box[1]-face_ex_h), 0))
    x2, y2 = int(min(math.ceil(box[2]+face_ex_w), w)), int(min(math.ceil(box[3]+face_ex_h), h))
    #print('x1, y1, x2, y2', x1, y1, x2, y2)
    img = img[y1:y2,x1:x2,:]
    point_meta.set_precrop_wh( img.shape[1], img.shape[0], x1, y1, x2, y2)
    point_meta.apply_offset(-x1, -y1)
    point_meta.apply_bound(img.shape[1], img.shape[0])

    return img, point_meta



class TrainScale2WH(object):

  # Rescale the input image to the given size.
 

  def __init__(self, target_size, interpolation=cv2.INTER_LINEAR):
    assert isinstance(target_size, tuple) or isinstance(target_size, list), 'The type of target_size is not right : {}'.format(target_size)
    assert len(target_size) == 2, 'The length of target_size is not right : {}'.format(target_size)
    assert isinstance(target_size[0], int) and isinstance(target_size[1], int), 'The type of target_size is not right : {}'.format(target_size)
    self.target_size   = target_size
    self.interpolation = interpolation

  def __call__(self, img, point_meta):
    """
    Args:
      img (cv.image): Image to be scaled.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
      cv.image: Rescaled image.
    """
    
 
    point_meta = point_meta.copy()

    h, w, _ = img.shape # h*w*c
    #print(img.shape)
    ow, oh = self.target_size[0], self.target_size[1]
    point_meta.apply_scale( [ow*1./w, oh*1./h] )

    # resize是 w*h*c
    img = cv2.resize(img, (ow, oh), self.interpolation)

    return img, point_meta


class AugScale(object):
    
  # Rescale the input image to the given size. Data Augmentation

  def __init__(self, scale_prob, scale_min, scale_max, interpolation=cv2.INTER_LINEAR):
    assert isinstance(scale_prob, numbers.Number) and scale_prob >= 0, 'scale_prob : {:}'.format(scale_prob)
    assert isinstance(scale_min,  numbers.Number) and isinstance(scale_max, numbers.Number), 'scales : {:}, {:}'.format(scale_min, scale_max)
    self.scale_prob = scale_prob
    self.scale_min  = scale_min
    self.scale_max  = scale_max
    self.interpolation = interpolation

  def __call__(self, img, point_meta):
    """
    Args:
      img (cv.image): Image to be scaled.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
      cv.image: Rescaled image.
    """
    
 
    point_meta = point_meta.copy()

    dice = random.random()
    if dice > self.scale_prob:
      return img, point_meta

    scale_multiplier = (self.scale_max - self.scale_min) * random.random() + self.scale_min
    
    h, w, _ = img.shape # h*w*c
    ow, oh = int(w * scale_multiplier), int(h * scale_multiplier)

    img = cv2.resize(img, (ow, oh), self.interpolation)
    point_meta.apply_scale( [scale_multiplier] )


    return img, point_meta

class AugTransBbox(object):
    
  # Random translations the bounding box. Data Augmentation
  
  def __init__(self, trans_prob, trans_percent):
    assert isinstance(trans_prob, numbers.Number) and trans_prob >= 0, 'trans_prob : {:}'.format(trans_prob)
    assert isinstance(trans_percent, numbers.Number) and trans_percent >= 0, 'trans_prob : {:}'.format(trans_percent)
    
    self.trans_prob = trans_prob
    self.trans_percent = trans_percent
     
  def __call__(self, img, point_meta):
    """
    Args:
      img (cv.image): image.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
      cv.image: Rescaled image.
    """
    
    point_meta = point_meta.copy()
    
    dice = random.random()
    if dice > self.trans_prob:
      return img, point_meta

    h, w, _ = img.shape  # h*w*c
    box = point_meta.get_box()
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    x1_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_w
    y1_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_h
    x2_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_w
    y2_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_h
       
    x1, y1 = int(max(math.floor(box[0]+x1_diff), 0)), int(max(math.floor(box[1]+y1_diff), 0))
    x2, y2 = int(min(math.ceil(box[2]+x2_diff), w)), int(min(math.ceil(box[3]+y2_diff), h))
    
    point_meta.set_box([x1, y1, x2, y2])

    return img, point_meta



class AugRotate(object):
  """Rotate the given cv.Image at the center.
  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, max_rotate_degree):
    assert isinstance(max_rotate_degree, numbers.Number)
    self.max_rotate_degree = max_rotate_degree

  def __call__(self, img, point_meta):
    """
    Args:
      img (cv.Image): Image to be rotated.
      point_meta : Point_Meta
    Returns:
      cv.Image: Rotated image.
    """
    

    point_meta = point_meta.copy()

    degree = (random.random() - 0.5) * 2 * self.max_rotate_degree
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, degree, 1)
    img = cv2.warpAffine(img, M, (w,h))

    point_meta.apply_rotate(center, degree)
    point_meta.apply_bound(img.shape[1], img.shape[0])

    return img, point_meta


class AugCrop(object):

  def __init__(self, crop_x, crop_y, center_perterb_max, fill=0):
    assert isinstance(crop_x, int) and isinstance(crop_y, int) and isinstance(center_perterb_max, numbers.Number)
    self.crop_x = crop_x
    self.crop_y = crop_y
    self.center_perterb_max = center_perterb_max
    assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
    self.fill   = fill

  def __call__(self, img, point_meta=None):
    ## AugCrop has something wrong... For unsupervised data
    ## 在框中心加干扰并且固定到一个定大小
    
    
    point_meta = point_meta.copy()
   

    dice_x, dice_y = random.random(), random.random()
    x_offset = int( (dice_x-0.5) * 2 * self.center_perterb_max)
    y_offset = int( (dice_y-0.5) * 2 * self.center_perterb_max)
    
    x1 = int(round( point_meta.center[0] + x_offset - self.crop_x / 2. ))
    y1 = int(round( point_meta.center[1] + y_offset - self.crop_y / 2. ))
    x2 = x1 + self.crop_x
    y2 = y1 + self.crop_y

    (h, w) = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
      pad = max(0-x1, 0-y1, x2-w+1, y2-h+1)
      assert pad > 0, 'padding operation in crop must be greater than 0'
      img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=self.fill)
      # 调整到新图像的原点坐标， 旧图像的原点在新图像中是(pad, pad)    
      x1, x2, y1, y2 = x1 + pad, x2 + pad, y1 + pad, y2 + pad
      point_meta.apply_offset(pad, pad)
      point_meta.apply_bound(img.shape[1], img.shape[0])
    
    # 正常的crop
    point_meta.apply_offset(-x1, -y1)
    
    #print(point_meta.box)
    #print(point_meta.center[0])
    #print(point_meta.center[1])
    #print(point_meta.image_path)
    #print('{} {} {} {}'.format(x1, y1, x2, y2))
    img = img[y1:y2,x1:x2,:]
    point_meta.apply_bound(img.shape[1], img.shape[0])

    
    return img, point_meta


class AugGaussianBlur(object):
  """Rotate the given cv.Image at the center.
  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, gaussianblur_prob,kernel_size, sigma):
    assert isinstance(kernel_size, numbers.Number)
    assert isinstance(sigma, numbers.Number)
    self.kernel_size = kernel_size
    self.sigma = sigma
    self.gaussianblur_prob = gaussianblur_prob

  def __call__(self, img, point_meta):
    """
    Args:
      img (cv.Image): Image to be rotated.
      point_meta : Point_Meta
    Returns:
      cv.Image: Rotated image.
    """
    

    point_meta = point_meta.copy()

    dice = random.random()
    if dice > self.gaussianblur_prob:
      return img, point_meta
    
    img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)

    return img, point_meta



class AugHorizontalFlip(object):
  """Rotate the given cv.Image at the center.
  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, flip_prob):
    assert isinstance(flip_prob, numbers.Number)   
    self.flip_prob = flip_prob

  def __call__(self, img, point_meta):
    """
    Args:
      img (cv.Image): Image to be rotated.
      point_meta : Point_Meta
    Returns:
      cv.Image: Rotated image.
    """
    
    point_meta = point_meta.copy()

    dice = random.random()
    if dice > self.flip_prob:
      return img, point_meta
    
    img = cv2.flip(img, 1)
    (h, w) = img.shape[:2]
    point_meta.apply_horizontal_flip(w)

    return img, point_meta



class ToTensor(object):
  """
  numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """
  def __call__(self, img, points):

 
    pic = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #opencv内置 numpy 先调整色彩通道，再调整维度
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    img = img.float().div(255)
      
    return img, points




