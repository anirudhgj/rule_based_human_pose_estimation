import tensorflow as tf
import numpy as np

from transform_util import root_relative_to_view_norm_skeleton
import procrustes

def unit_norm(mat, dim=2):
    norm = (np.sqrt(np.sum(mat ** 2, dim)) + 1e-9)
    norm = np.expand_dims(norm, dim)
    mat = mat / norm
    return mat

def get_distance(point1, point2):
    x1,y1,z1 = point1[:,0], point1[:,1], point1[:,2]
    x2,y2,z2 = point2[:,0], point2[:,1], point2[:,2]
    dist = np.sqrt(np.square(x1-x2) + np.square(y1-y2) + np.square(z1-z2))
    
    return np.expand_dims(dist, 1)

def get_gt_bone_lengths(pred_3d, gt_3d):
    
    pelvis = pred_3d[:,0,:]
    rhip = unit_norm(pred_3d[:,9,:], dim=1) * get_distance(gt_3d[:,0], gt_3d[:,9]) + pelvis
    lhip = unit_norm(pred_3d[:,13,:], dim=1) * get_distance(gt_3d[:,0], gt_3d[:,13]) + pelvis
    neck = unit_norm(pred_3d[:,1,:], dim=1) * get_distance(gt_3d[:,0], gt_3d[:,1]) + pelvis
    rs = unit_norm(pred_3d[:,2,:]-pred_3d[:,1,:], dim=1) * get_distance(gt_3d[:,1], gt_3d[:,2]) + neck
    re = unit_norm(pred_3d[:,3,:]-pred_3d[:,2,:], dim=1) * get_distance(gt_3d[:,2], gt_3d[:,3]) + rs
    rh = unit_norm(pred_3d[:,4,:]-pred_3d[:,3,:], dim=1) * get_distance(gt_3d[:,3], gt_3d[:,4]) + re
    ls = unit_norm(pred_3d[:,5,:]-pred_3d[:,1,:], dim=1) * get_distance(gt_3d[:,1], gt_3d[:,5]) + neck
    le = unit_norm(pred_3d[:,6,:]-pred_3d[:,5,:], dim=1) * get_distance(gt_3d[:,5], gt_3d[:,6]) + ls
    lh = unit_norm(pred_3d[:,7,:]-pred_3d[:,6,:], dim=1) * get_distance(gt_3d[:,6], gt_3d[:,7]) + le
    head = unit_norm(pred_3d[:,8,:]-pred_3d[:,1,:], dim=1) * get_distance(gt_3d[:,1], gt_3d[:,8]) + neck
    rk = unit_norm(pred_3d[:,10,:]-pred_3d[:,9,:], dim=1) * get_distance(gt_3d[:,9], gt_3d[:,10]) + rhip
    ra = unit_norm(pred_3d[:,11,:]-pred_3d[:,10,:], dim=1) * get_distance(gt_3d[:,10], gt_3d[:,11]) + rk
    rf = unit_norm(pred_3d[:,12,:]-pred_3d[:,11,:], dim=1) * get_distance(gt_3d[:,11], gt_3d[:,12]) + ra
    lk = unit_norm(pred_3d[:,14,:]-pred_3d[:,13,:], dim=1) * get_distance(gt_3d[:,13], gt_3d[:,14]) + lhip
    la = unit_norm(pred_3d[:,15,:]-pred_3d[:,14,:], dim=1) * get_distance(gt_3d[:,14], gt_3d[:,15]) + lk
    lf = unit_norm(pred_3d[:,16,:]-pred_3d[:,15,:], dim=1) * get_distance(gt_3d[:,15], gt_3d[:,16]) + la
    
    skeleton = np.concatenate([np.expand_dims(pelvis, axis=1), np.expand_dims(neck, axis=1), 
                            np.expand_dims(rs, axis=1), np.expand_dims(re, axis=1),
                            np.expand_dims(rh, axis=1), np.expand_dims(ls, axis=1),
                            np.expand_dims(le, axis=1), np.expand_dims(lh, axis=1),
                            np.expand_dims(head, axis=1), np.expand_dims(rhip, axis=1),
                            np.expand_dims(rk, axis=1), np.expand_dims(ra, axis=1),
                            np.expand_dims(rf, axis=1), np.expand_dims(lhip, axis=1), 
                            np.expand_dims(lk, axis=1), np.expand_dims(la, axis=1),
                            np.expand_dims(lf, axis=1)], axis=1)
    return skeleton

def compute_mpjpe_summary(gt_pose, pred_pose):
    """
    Returns the (PA-) MPJPE of a batch of predicted poses using GT poses.
    gt_pose: A Bx17x3 array of predicted 3D poses
    pred_pose: A Bx17x3 array of GT 3D poses.
    Reference: https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/predict_3dpose.py
    """
    pred_pose = get_gt_bone_lengths(pred_pose.copy(), gt_pose.copy())


    pa_pose = pred_pose.copy()
    for i in range(gt_pose.shape[0]):
        gt = gt_pose[i]
        out = pred_pose[i]
        _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
        out = (b*out.dot(T))+c
        pa_pose[i, :, :] = out
    return np.mean(np.sqrt(np.sum((pa_pose - gt_pose)**2, axis = 2)))


