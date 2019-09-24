


def box2square(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    
    center_x = (x1 + x2)/2
    center_y = (y1 + y2)/2
    
    w = x2 - x1
    h = y2 - y1
    
    s_len = max(w, h)
    
    new_box = [0, 0, 0, 0]
    new_box[0], new_box[1], new_box[2], new_box[3] = \
    center_x - s_len/2, center_y - s_len/2, center_x + s_len/2, center_y + s_len/2
    
    return new_box