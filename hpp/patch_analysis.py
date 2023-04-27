"""
Patch classifiers
"""
import time
import sys

from numpy.core.records import array
import cv2
import numpy as np
from hpp import common
from hpp import selection as sel
from hpp import run_config as cfg


def aux_comp_soft_segs_with_clasif(all_imgs_color_code, selection, weights):
    """
    Classify color patches based on a provided model

    [in] all_imgs_color_code - array with color codifications for all images
    [in] selection - indicates selected features
    [in] weights - weights of the considered linear model

    [out] all_soft_segs - array with all soft segs
    """
    n_rows = all_imgs_color_code.shape[0]
    n_cols = all_imgs_color_code.shape[1]

    x_coords, y_coords = aux_get_patches_locations(
        n_cols, n_rows, cfg.PATCH_SOFT_SEG_TEST_SAMPLES_STEP_X,
        cfg.PATCH_SOFT_SEG_TEST_SAMPLES_STEP_Y)

    im_col_aux = np.zeros(all_imgs_color_code.shape, np.int32)
    selected_colors = np.nonzero(selection)
    selected_colors = selected_colors[0]
    selected_colors = selected_colors[1:]

    for i in range(0, np.prod(selected_colors.shape)):
        pos = all_imgs_color_code == selected_colors[i]
        im_col_aux[pos] = i + 1

    all_soft_segs = np.zeros((n_rows, n_cols))
    features = np.zeros((np.sum(selection), 1))

    for i in range(0, np.prod(x_coords.shape)):
        features = 0 * features
        img_patch = im_col_aux[
            y_coords[i] - cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE :\
            y_coords[i] + cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE + 1,
            x_coords[i] - cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE :\
            x_coords[i] + cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE + 1]
        pos = img_patch > 0

        features[img_patch[pos]] = 1
        features[0] = 1
        all_soft_segs[y_coords[i],
                      x_coords[i]] = np.transpose(features).dot(weights)

    all_soft_segs = common.normalize_tensor(all_soft_segs)

    return all_soft_segs


def aux_get_img_color_code(img_hsv):
    """
    Get image color codification

    [in] img_hsv - image in hsv color space

    [out] img_color_code - color codification
    """
    h_channel = common.normalize_tensor(img_hsv[:, :, 0])
    s_channel = common.normalize_tensor(img_hsv[:, :, 1])
    v_channel = common.normalize_tensor(img_hsv[:, :, 2])

    h_channel = np.int32(
        np.round(h_channel * (cfg.PATCH_SOFT_SEG_DIM_H - 1) + 1))
    s_channel = np.int32(
        np.round(s_channel * (cfg.PATCH_SOFT_SEG_DIM_S - 1) + 1))
    v_channel = np.int32(
        np.round(v_channel * (cfg.PATCH_SOFT_SEG_DIM_V - 1) + 1))

    img_color_code = h_channel\
        + (s_channel - 1) * cfg.PATCH_SOFT_SEG_DIM_H\
        + (v_channel - 1) * cfg.PATCH_SOFT_SEG_DIM_H * cfg.PATCH_SOFT_SEG_DIM_S

    return img_color_code


def aux_get_patches_locations(n_cols, n_rows, x_step, y_step):
    """
    Get locations of patches considered for training or testing

    [in] n_cols - n cols in the considered image
    [in] n_rows - n rows in the considered image
    [in] x_step - step between two consecutive patches - x dim
    [in] y_step - step between two consecutive patches - y dim

    [out] x_coords - x coords
    [out] y_coords - y coords
    """
    x_coords0 = np.arange(cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE,
                          n_cols - cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE, x_step)
    x_coords0 = x_coords0[:, np.newaxis]
    x_coords0 = np.transpose(x_coords0)
    n_x_coords = np.prod(x_coords0.shape)

    y_coords0 = np.arange(cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE,
                          n_rows - cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE, y_step)
    y_coords0 = y_coords0[:, np.newaxis]
    n_y_coords = np.prod(y_coords0.shape)

    x_coords = np.repeat(x_coords0, n_y_coords, axis=0)
    y_coords = np.repeat(y_coords0, n_x_coords, axis=1)

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    return x_coords, y_coords


def aux_get_model(all_imgs_color_code, all_soft_segs):
    """
    Select features and compute classification model

    [in] all_imgs_color_code - array with color codifications of all images
    [in] all_soft_segs - all soft_segs

    [out] selection - binary mask of selected features
    [out] weights - computed weights
    """
    n_rows = all_imgs_color_code.shape[0]
    n_cols = all_imgs_color_code.shape[1]

    x_coords, y_coords = aux_get_patches_locations(
        n_cols, n_rows, cfg.PATCH_SOFT_SEG_TRAIN_SAMPLES_STEP_X,
        cfg.PATCH_SOFT_SEG_TRAIN_SAMPLES_STEP_Y)

    n_patches = np.prod(x_coords.shape)

    samples = np.zeros(\
        (n_patches,\
        1 + cfg.PATCH_SOFT_SEG_DIM_H * cfg.PATCH_SOFT_SEG_DIM_S * cfg.PATCH_SOFT_SEG_DIM_V))
    targets = np.zeros((n_patches, 1))

    samples[:, 0] = 1

    for i in range(0, n_patches):
        colors = all_imgs_color_code[\
            y_coords[i] - cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE :\
            y_coords[i] + cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE + 1,\
            x_coords[i] - cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE :\
            x_coords[i] + cfg.PATCH_SOFT_SEG_HALF_PATCH_SIZE + 1]
        targets[i] = all_soft_segs[y_coords[i], x_coords[i]]
        colors = colors.flatten()
        samples[i, colors] = 1

    color_samples = samples[:,\
        1 : (1 + cfg.PATCH_SOFT_SEG_DIM_H * cfg.PATCH_SOFT_SEG_DIM_S * cfg.PATCH_SOFT_SEG_DIM_V)]
    color_samples = color_samples - np.mean(color_samples, axis=0)


    selection = sel.cluster_ipfp(\
        np.transpose(color_samples).dot(color_samples),\
        cfg.PATCH_SOFT_SEG_N_FEATURES)
    selection = selection > 0
    selection = np.concatenate((np.array([[True]]), selection))

    samples = samples[:, np.squeeze(selection)]
    n_features = np.sum(selection)

    weights = ((np.linalg.inv(np.transpose(samples).dot(samples)\
        + cfg.PATCH_SOFT_SEG_LAMBDA * np.identity(n_features)))\
        .dot(np.transpose(samples)))\
        .dot(targets)

    return selection, weights


def aux_classify_color_patches_ipfp(all_imgs_hsv, all_soft_segs):
    """
    Classify color patches

    [in] all_imgs_hsv - array with all images in HSV color space
    [in] all_soft_segs - array with all soft segs

    [out] patch_based_all_soft_segs - array with all patch based soft segs
    """

    all_imgs_color_code = aux_get_img_color_code(all_imgs_hsv)

    selection, weights = aux_get_model(all_imgs_color_code, all_soft_segs)

    patch_based_all_soft_segs =\
        aux_comp_soft_segs_with_clasif(all_imgs_color_code, selection, weights)

    return patch_based_all_soft_segs


def aux_prepare_imgs_and_fg_ques(frames, soft_segs):
    """
    Pre-process frames and soft segs to prepare them for patch analysis
    (Note that both frames and soft segs are downsampled to half their size)

    [in] frames - list of frames - BGR color space
    [in] soft_segs - list of considered soft segmentations

    [out] all_frames - an array containing all frames info in hsv color space
    [out] all_soft_segs - an array containing all soft segmentations after filtering process
    """
    n_frames = len(frames)
    n_rows_half = np.int32(frames[0].shape[0] * 0.5)
    n_cols_half = np.int32(frames[0].shape[1] * 0.5)

    central_gaussian = common.gauss_kernel(
        shape=(n_rows_half, n_cols_half),
        sigma=min(n_rows_half, n_cols_half) / 3)

    all_frames = np.zeros((n_rows_half, n_cols_half * n_frames, 3))
    all_soft_segs = np.zeros((n_rows_half, n_cols_half * n_frames))

    last_column = 0

    for idx, frame in enumerate(frames):
        frame = cv2.resize(  #pylint:disable=no-member
            common.normalize_tensor(frame),
            dsize=(n_cols_half, n_rows_half),
            interpolation=cv2.INTER_CUBIC)  #pylint:disable=no-member
        frame = common.normalize_tensor(frame)
        hsv = cv2.cvtColor(np.uint8(frame * 255), cv2.COLOR_BGR2HSV)  #pylint:disable=no-member

        all_frames[:, last_column:last_column + n_cols_half,
                   0] = hsv[:, :, 0] * 2.0
        all_frames[:, last_column:last_column + n_cols_half,
                   1] = hsv[:, :, 1] / 255.0
        all_frames[:, last_column:last_column + n_cols_half,
                   2] = hsv[:, :, 2] / 255.0

        soft_seg = soft_segs[idx]
        soft_seg = cv2.resize(  #pylint:disable=no-member
            soft_seg,
            dsize=(n_cols_half, n_rows_half),
            interpolation=cv2.INTER_CUBIC)  #pylint:disable=no-member
        soft_seg = common.normalize_tensor(soft_seg)
        soft_seg = common.normalize_tensor(soft_seg * common.hysthresh(
            common.normalize_tensor(soft_seg * central_gaussian),
            cfg.PATCH_SOFT_SEG_HYSTHRESH_T1, cfg.PATCH_SOFT_SEG_HYSTHRESH_T2))
        all_soft_segs[:, last_column:last_column + n_cols_half] = soft_seg

        last_column = last_column + n_cols_half

    return all_frames, all_soft_segs


def get_patch_based_soft_seg(frames, soft_segs):
    """
    Computed patch based soft-segmentations for a set of frames for which
    we have a set of foreground ques

    [in] frames - list of considered frames
    [in] soft_segs - list of estimated soft-segs - used as foreground ques

    [out] patch_soft_segs - computed patch based soft segs
    """
    all_frames_hsv, all_soft_segs = aux_prepare_imgs_and_fg_ques(
        frames, soft_segs)

    patch_based_all_soft_segs = aux_classify_color_patches_ipfp(
        all_frames_hsv, all_soft_segs)

    n_rows = frames[0].shape[0]
    n_cols = frames[0].shape[1]
    n_rows_half = np.int32(n_rows * 0.5)
    n_cols_half = np.int32(n_cols * 0.5)

    min_value = sys.float_info.max
    max_value = sys.float_info.min

    patch_soft_segs = []

    for i in range(0, len(frames)):
        patch_based_soft_seg = patch_based_all_soft_segs[:, i *
                                                         n_cols_half:(i + 1) *
                                                         n_cols_half]

        soft_seg = soft_segs[i]
        soft_seg = cv2.resize(#pylint:disable=no-member
            soft_seg,\
            dsize=(n_cols_half, n_rows_half),\
            interpolation=cv2.INTER_CUBIC)#pylint:disable=no-member
        soft_seg = common.normalize_tensor(soft_seg)

        soft_seg = patch_based_soft_seg * soft_seg

        soft_seg = cv2.resize(soft_seg,
                              dsize=(n_cols, n_rows),
                              interpolation=cv2.INTER_CUBIC)  #pylint:disable=no-member
        soft_seg[soft_seg < 0] = 0
        patch_soft_segs.append(soft_seg)
        min_value = min(min_value, np.min(soft_seg))
        max_value = max(max_value, np.max(soft_seg))

    for i in range(0, len(frames)):
        soft_seg = patch_soft_segs[i]
        soft_seg = (soft_seg - min_value) / (max_value - min_value)
        patch_soft_segs[i] = soft_seg

   
    return patch_soft_segs
