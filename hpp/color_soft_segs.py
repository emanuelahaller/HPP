"""
Compute soft-segmentation masks based on pixel colors
"""
import sys
import math
import numpy as np
from hpp import run_config as cfg


def aux_get_color_codifications(img_hsv, binary_mask):
    """
    Get pixel color codifications based on combination of HSV values
    The values are compute only for indicated pixels

    [in] img_hsv - considered image in HSV color space
    [in] binary_mask - binary mask of pixels that need to be considered
    [out] col - color codifications
    """

    h_values = img_hsv[:, :, 0]
    s_values = img_hsv[:, :, 1]
    v_values = img_hsv[:, :, 2]

    h_values = h_values[binary_mask]
    s_values = s_values[binary_mask]
    v_values = v_values[binary_mask]

    h_values_aux = (s_values * np.cos((h_values / 180) * math.pi) + 1) / 2
    s_values_aux = (s_values * np.sin((h_values / 180) * math.pi) + 1) / 2

    h_values = h_values_aux
    s_values = s_values_aux

    h_values = np.int32(np.round(h_values * (cfg.COLOR_SOFT_SEG_DIM_H - 1) +
                                 1))
    v_values = np.int32(np.round(v_values * (cfg.COLOR_SOFT_SEG_DIM_V - 1) +
                                 1))
    s_values = np.int32(np.round(s_values * (cfg.COLOR_SOFT_SEG_DIM_S - 1) +
                                 1))

    col = h_values
    col = col + (s_values - 1) * cfg.COLOR_SOFT_SEG_DIM_H
    col = col + (v_values -
                 1) * cfg.COLOR_SOFT_SEG_DIM_H * cfg.COLOR_SOFT_SEG_DIM_S

    return col


def aux_get_color_histogram(img_hsv, binary_mask):
    """
    Compute color histogram for pixels indicated by the binary mask

    [in] img_hsv - considered image in HSV color space
    [in] binary_mask - binary mask of pixels that need to be considered for histogram

    [out] hist_col - color histogram
    [out] col - color codification
    """
    col = aux_get_color_codifications(img_hsv, binary_mask)

    hist_col = col
    n_bins = cfg.COLOR_SOFT_SEG_DIM_H * cfg.COLOR_SOFT_SEG_DIM_S * cfg.COLOR_SOFT_SEG_DIM_V
    hist_col, _ = np.histogram(hist_col, np.arange(1, n_bins + 2))
    hist_col = hist_col[0:n_bins]
    hist_col = hist_col / (np.sum(hist_col) + sys.float_info.epsilon)

    return hist_col, col


def get_color_soft_seg(img_hsv, binary_mask):
    """
    Compute color based soft segmentation of an image based on provided foreground ques

    [in] img_hsv - image in HSV color space
    [in] binary_mask - considered binary mask containing foreground ques

    [out] color_soft_seg - computed color based soft segmentation
    """
    n_rows = img_hsv.shape[0]
    n_cols = img_hsv.shape[1]

    foreground_mask = binary_mask > 0
    background_mask = binary_mask == 0

    foreground_hist_col, foreground_col = aux_get_color_histogram(
        img_hsv, foreground_mask)
    background_hist_col, background_col = aux_get_color_histogram(
        img_hsv, background_mask)

    color_soft_seg = np.zeros((n_rows, n_cols))

    color_soft_seg[foreground_mask] = foreground_hist_col[foreground_col - 1] / \
        (foreground_hist_col[foreground_col - 1] + background_hist_col[foreground_col - 1])
    color_soft_seg[background_mask] = foreground_hist_col[background_col - 1] / \
        (foreground_hist_col[background_col - 1] + background_hist_col[background_col - 1])

    return color_soft_seg
