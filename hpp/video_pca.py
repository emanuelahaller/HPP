"""
VideoPCA related algorithms
"""
import cv2
import time
import numpy as np
from hpp import common
from hpp import run_config as cfg
from hpp import color_soft_segs


def aux_get_color_soft_seg(frames, fg_ques):
    """
    Compute color soft segmentation based on the provided foreground ques

    [in] frames - list of frames
    [in] fg_ques - list of fg ques - soft segs

    [out] soft_segs - computed soft segmentations
    """
    n_frames = len(frames)
    n_rows, n_cols, n_channels = frames[0].shape

    all_imgs_hsv = np.zeros((n_rows, n_cols * n_frames, n_channels))
    all_binary_masks = np.zeros((n_rows, n_cols * n_frames))

    blur_kernel = common.gauss_kernel(\
        shape=(3 * cfg.VIDEO_PCA_SIGMA_BLUR, 3 * cfg.VIDEO_PCA_SIGMA_BLUR),\
        sigma=cfg.VIDEO_PCA_SIGMA_BLUR)
    central_gauss = common.gauss_kernel(\
        shape=(n_rows, n_cols),\
        sigma=min(n_rows, n_cols)/3)

    last = 0
    for i in range(0, n_frames):
        aux = fg_ques[i, :, :]
        aux = cv2.filter2D(aux, ddepth=-1, kernel=blur_kernel)  #pylint:disable=no-member
        aux = common.normalize_tensor(aux * central_gauss)

        hsv = cv2.cvtColor(np.uint8(frames[i] * 255), cv2.COLOR_BGR2HSV)  #pylint:disable=no-member

        all_imgs_hsv[:, last:last + n_cols, 0] = hsv[:, :, 0] * 2.0
        all_imgs_hsv[:, last:last + n_cols, 1] = hsv[:, :, 1] / 255.0
        all_imgs_hsv[:, last:last + n_cols, 2] = hsv[:, :, 2] / 255.0

        aux = common.normalize_tensor(aux)
        aux = aux > 0.5
        all_binary_masks[:, last:last + n_cols] = aux

        last = last + n_cols

    all_soft_segs = color_soft_segs.get_color_soft_seg(all_imgs_hsv,
                                                       all_binary_masks)

    soft_segs = []
    for i in range(0, n_frames):
        soft_seg = all_soft_segs[:, i * n_cols:(i + 1) * n_cols]
        aux = common.normalize_tensor(soft_seg * central_gauss)
        aux = common.hysthresh(aux, 0.8, 0.5)
        soft_seg = common.normalize_tensor(aux * soft_seg)
        soft_segs.append(soft_seg)

    return soft_segs


def aux_get_color_soft_seg_no_blur(frames, fg_ques):
    """
    Compute color soft segmentation based on the provided foreground ques

    [in] frames - list of frames
    [in] fg_ques - list of fg ques - soft segs

    [out] soft_segs - computed soft segmentations
    """
    n_frames = len(frames)
    n_rows, n_cols, n_channels = frames[0].shape

    all_imgs_hsv = np.zeros((n_rows, n_cols * n_frames, n_channels))
    all_binary_masks = np.zeros((n_rows, n_cols * n_frames))

    central_gauss = common.gauss_kernel(\
        shape=(n_rows, n_cols),\
        sigma=min(n_rows, n_cols)/3)

    last = 0
    for i in range(0, n_frames):
        aux = fg_ques[i, :, :]
        aux = common.normalize_tensor(aux * central_gauss)

        hsv = cv2.cvtColor(np.uint8(common.normalize_tensor(frames[i]) * 255),
                           cv2.COLOR_BGR2HSV)  #pylint:disable=no-member

        all_imgs_hsv[:, last:last + n_cols, 0] = hsv[:, :, 0] * 2.0
        all_imgs_hsv[:, last:last + n_cols, 1] = hsv[:, :, 1] / 255.0
        all_imgs_hsv[:, last:last + n_cols, 2] = hsv[:, :, 2] / 255.0

        binary_mask = common.normalize_tensor(aux)
        binary_mask = binary_mask > 0.5
        all_binary_masks[:, last:last + n_cols] = binary_mask

        last = last + n_cols

    all_soft_segs = color_soft_segs.get_color_soft_seg(all_imgs_hsv,
                                                       all_binary_masks)

    soft_segs = []
    for i in range(0, n_frames):
        soft_seg = all_soft_segs[:, i * n_cols:(i + 1) * n_cols]
        soft_segs.append(soft_seg)

    return soft_segs


def aux_apply_pca(data, n_dirs):
    """
    Apply pca algorithm considering the provided set of samples

    [in] data - data matrix
    [in] n_dirs - desired number of directions

    [out] eigenvectors - selected eigenvectors
    [out] mean_data - mean data
    """
    mean_data = np.mean(data, axis=0)
    data = data - mean_data

    [eigenvalues, eigenvectors] = np.linalg.eig(np.transpose(data).dot(data))
    ord_indices = np.argsort(eigenvalues)
    n_max_dirs = np.prod(eigenvalues.shape)
    n_dirs = min(n_dirs, n_max_dirs)
    sel_indices = np.arange(n_max_dirs - n_dirs, n_max_dirs)
    sel_indices = ord_indices[sel_indices]
    eigenvectors = eigenvectors[:, sel_indices]

    return eigenvectors, mean_data, data


def video_pca(frames):
    """
    Apply videoPCA algorithm

    [in] frames - list of frames

    [out] soft_segs - list of soft segs
    """
    
    n_frames = len(frames)
    n_rows, n_cols, n_channels = frames[0].shape

    data = np.zeros((n_frames, n_rows * n_cols * n_channels), np.float32)
    for idx in range(0, n_frames):
        frames[idx] = common.normalize_tensor(frames[idx])
        data[idx, :] = frames[idx].flatten()

    data = np.transpose(data)
    eigenvectors, mean_data, data = aux_apply_pca(data,
                                                  cfg.VIDEO_PCA_N_DIRECTIONS)

    rec_frames = np.dot(np.dot(data, eigenvectors),
                        np.transpose(eigenvectors)) + mean_data
    rec_frames_ = []
    rec_diff_frames_ = []

    rec_diff = np.zeros((n_frames, n_rows, n_cols), np.float32)

    for idx in range(0, n_frames):
        rec_frame = rec_frames[:, idx]
        rec_frame = np.reshape(rec_frame, (n_rows, n_cols, n_channels))
        rec_frames_.append(rec_frame)

        aux = frames[idx] - rec_frame
        aux = np.sqrt(np.sum(pow(aux, 2), axis=2))

        rec_diff[idx, :, :] = aux

    rec_diff = common.normalize_tensor(rec_diff)
    for i in range(n_frames):
        rec_diff_frames_.append(rec_diff[i, :, :])
    
    soft_segs = aux_get_color_soft_seg(frames, rec_diff)
    

    return soft_segs, rec_frames_, rec_diff_frames_

def video_pca_soft_segs(frames, soft_segs):
    """
    Apply videoPCA alg to previous soft segs as refinement

    [in] frames - list of frames
    [in] soft_segs - list of soft segmentations

    [out] refined_soft_segs - list of refined soft segmentations
    """
   
    n_frames = len(frames)
    n_rows = frames[0].shape[0]
    n_cols = frames[0].shape[1]

    data = np.zeros((n_frames, n_rows * n_cols), np.float32)

    for idx in range(0, n_frames):
        soft_seg = common.normalize_tensor(soft_segs[idx])
        data[idx, :] = soft_seg.flatten()

    data = np.transpose(data)
    eigenvectors, mean_data, data = aux_apply_pca(
        data, cfg.VIDEO_PCA_SOFT_SEGS_N_DIRECTIONS)

    rec_frames = np.dot(np.dot(data, eigenvectors),
                        np.transpose(eigenvectors)) + mean_data
    rec_diff = np.zeros((n_frames, n_rows, n_cols), np.float32)
    rec_frames_ = []
    for idx in range(0, n_frames):
        rec_frame = rec_frames[:, idx]
        rec_frame = np.reshape(rec_frame, (n_rows, n_cols))
        rec_frames_.append(rec_frame)
        rec_diff[idx, :, :] = common.normalize_tensor(rec_frame)

    rec_diff = common.normalize_tensor(rec_diff)
    for i in range(n_frames):
        rec_frames_.append(rec_diff[i, :, :])
    
    soft_segs = aux_get_color_soft_seg_no_blur(frames, rec_diff)
    
    return soft_segs, rec_frames_