import torch
import sys, math
import numpy as np
import numbers
import cv2


class Normalize(object):
  """Normalize an tensor image with mean and standard deviation.
  Given mean: (B, G, R) and std: (B, G, R),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  Args:
    mean (sequence): Sequence of means for B, G, R channels respecitvely.
    std (sequence): Sequence of standard deviations for B, G, R channels
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

  def __call__(self, imgs, point_meta):
    ## AugCrop has something wrong... For unsupervised data

    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]


    h, w, _ = imgs[0].shape  # h*w*c
    box = point_meta.get_box().tolist()
    face_ex_w, face_ex_h = (box[2] - box[0]) * self.expand_ratio, (box[3] - box[1]) * self.expand_ratio
    x1, y1 = int(max(math.floor(box[0]-face_ex_w), 0)), int(max(math.floor(box[1]-face_ex_h), 0))
    x2, y2 = int(min(math.ceil(box[2]+face_ex_w), w)), int(min(math.ceil(box[3]+face_ex_h), h))

    imgs = [ img[y1:y2,x1:x2,:] for img in imgs ]
    point_meta.set_precrop_wh( imgs[0].shape[1], imgs[0].shape[0], x1, y1, x2, y2)
    point_meta.apply_offset(-x1, -y1)
    point_meta.apply_bound(imgs[0].shape[1], imgs[0].shape[0])

    if is_list == False: imgs = imgs[0]
    return imgs, point_meta



class TrainScale2WH(object):

  # Rescale the input image to the given size.
 

  def __init__(self, target_size, interpolation=cv2.INTER_LINEAR):
    assert isinstance(target_size, tuple) or isinstance(target_size, list), 'The type of target_size is not right : {}'.format(target_size)
    assert len(target_size) == 2, 'The length of target_size is not right : {}'.format(target_size)
    assert isinstance(target_size[0], int) and isinstance(target_size[1], int), 'The type of target_size is not right : {}'.format(target_size)
    self.target_size   = target_size
    self.interpolation = interpolation

  def __call__(self, imgs, point_meta):
    """
    Args:
      img (cv.image): Image to be scaled.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
      cv.image: Rescaled image.
    """
    point_meta = point_meta.copy()

    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]

    h, w, _ = imgs[0].shape # h*w*c
    ow, oh = self.target_size[0], self.target_size[1]
    point_meta.apply_scale( [ow*1./w, oh*1./h] )

    # resize是 w*h*c
    imgs = [ cv2.resize(img, (ow, oh), self.interpolation) for img in imgs ]
    if is_list == False: imgs = imgs[0]

    return imgs, point_meta


class AugScale(object):
    
  # Rescale the input image to the given size. Data Augmentation

  def __init__(self, scale_prob, scale_min, scale_max, interpolation=cv2.BILINEAR):
    assert isinstance(scale_prob, numbers.Number) and scale_prob >= 0, 'scale_prob : {:}'.format(scale_prob)
    assert isinstance(scale_min,  numbers.Number) and isinstance(scale_max, numbers.Number), 'scales : {:}, {:}'.format(scale_min, scale_max)
    self.scale_prob = scale_prob
    self.scale_min  = scale_min
    self.scale_max  = scale_max
    self.interpolation = interpolation

  def __call__(self, imgs, point_meta):
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
      return imgs, point_meta

    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]
    
    scale_multiplier = (self.scale_max - self.scale_min) * random.random() + self.scale_min
    
    h, w, _ = imgs[0].shape # h*w*c
    ow, oh = int(w * scale_multiplier), int(h * scale_multiplier)

    imgs = [ cv2.resize(img, (ow, oh), self.interpolation) for img in imgs ]
    point_meta.apply_scale( [scale_multiplier] )

    if is_list == False: imgs = imgs[0]

    return imgs, point_meta

class AugTransBbox(object):
    
  # Random translations the bounding box. Data Augmentation
  
  def __init__(self, trans_prob, trans_percent):
    assert isinstance(trans_prob, numbers.Number) and trans_prob >= 0, 'trans_prob : {:}'.format(trans_prob)
    assert isinstance(trans_percent, numbers.Number) and trans_percent >= 0, 'trans_prob : {:}'.format(trans_percent)
    
    self.trans_prob = trans_prob
    self.trans_percent = trans_percent
     
  def __call__(self, imgs, point_meta):
    """
    Args:
      img (cv.image): image.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
      cv.image: Rescaled image.
    """
    
    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]
    
    point_meta = point_meta.copy()
    
    dice = random.random()
    if dice > self.trans_prob:
      return imgs, point_meta

    h, w, _ = imgs[0].shape  # h*w*c
    box = point_meta.get_box().tolist()
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    x1_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_w
    y1_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_h
    x2_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_w
    y2_diff = (2*self.trans_percent*random.random() - self.trans_percent)*box_h
       
    x1, y1 = int(max(math.floor(box[0]+x1_diff), 0)), int(max(math.floor(box[1]+y1_diff), 0))
    x2, y2 = int(min(math.ceil(box[2]+x2_diff), w)), int(min(math.ceil(box[3]+y2_diff), h))
    
    point_meta.set_box([x1, y1, x2, y2])
    
    if is_list == False: imgs = imgs[0]
    return imgs, point_meta





