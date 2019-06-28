import numpy as np



def compute_nme(num_pts, predictions, groundtruth):
    num_images = len(predictions)
    error_per_image = np.zeros((num_images,1))

    for idx in range(num_images):
        det_pts = predictions[idx]
        gt_pts = groundtruth[idx]

        if num_pts == 81:
            interocular_d = np.linalg.norm(gt_pts[2*49:2*49+2]-gt_pts[2*62:2*62+2])
        if num_pts == 68:
            interocular_d = np.linalg.norm(gt_pts[2*36:2*36+2]-gt_pts[2*45:2*45+2])
        #print(interocular_d)
        
        dis_sum = 0
        for i in range(68):
            dis_sum = dis_sum + np.linalg.norm(det_pts[2*i:2*i+2] - gt_pts[2*i:2*i+2])
        error_per_image[idx] = dis_sum / (num_pts* interocular_d)
       
    return error_per_image.mean()
