import numpy as np



def compute_nme(nums_pt, prediction, gt):
    num_images = len(predictions)
    error_per_image = np.zeros((num_images,1))

    for idx in range(num_images):
        det_pts = predictions[idx]
        gt_pts = groundtruth[idx]
        if nums_pt == 81:
            interocular_d = np.linalg.norm(gt_pts[2*49:2*49+2]-gt_pts[2*62:2*62+2])

        dis_sum = 0
        for j in range(num_pts):
            dis_sum = dis_sum + np.linalg.norm(det_pts[:]-gt_pts[:])
        error_per_image[idx] = dis_sum / (num_pts* interocular_d)
    return error_per_image.mean()
