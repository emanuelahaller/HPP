"""
Useful functions for hpp vos method
"""
import sys
import numpy as np
import cv2


def normalize_tensor(tens):
    """
    Normalize a tensor

    [in] - tens - considered tensor
    [out] - normalized tensor - range [0,1]
    """

    tens = np.float32(tens)
    tens = tens - np.min(tens)
    tens = tens / (np.max(tens) + sys.float_info.epsilon)

    return tens


def gauss_kernel_with_center(shape=(3, 3), sigma=0.5, center=(1, 1)):
    """
    Generates a gaussian kernel that can further be applied to an image
    (centered in a given location)

    [in] shape - desired kernel size (img size)
    [in] sigma - gaussian param
    [in] center - where the gaussian is centered

    [out] generated kernel
    """
    ygrid, xgrid = np.ogrid[-center[0]:shape[0] - center[0],
                            -center[1]:shape[1] - center[1]]
    kernel = np.exp(-(xgrid * xgrid + ygrid * ygrid) / (2. * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel /= kernel_sum
    return kernel


def gauss_kernel(shape=(3, 3), sigma=0.5):
    """
    Generates a gaussian kernel that can further be applied to an image

    [in] shape - desired kernel size (img size)
    [in] sigma - gaussian param

    [out] generated kernel
    """
    xcenter = (shape[1] - 1.0) / 2.0
    ycenter = (shape[0] - 1.0) / 2.0
    ygrid, xgrid = np.ogrid[-ycenter:ycenter + 1, -xcenter:xcenter + 1]
    kernel = np.exp(-(xgrid * xgrid + ygrid * ygrid) / (2. * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel /= kernel_sum
    return kernel


def hysthresh(soft_seg, th1, th2):
    """
    Apply hysthresh to a soft seg => binary mask

    [in] soft_seg - considered soft segmentation
    [in] th1 - first threshold
    [in] th2 - second threshold

    [out] generated binary mask
    """

    if th1 < th2:
        tmp = th1
        th1 = th2
        th2 = tmp

    above_t2 = soft_seg > th2
    above_t1 = soft_seg > th1

    n_components, blobs = cv2.connectedComponents(np.uint8(above_t2), 8,
                                                  cv2.CV_32S)  #pylint:disable=no-member
    sel_components = set(blobs[above_t1])
    final_mask = above_t1 * 0
    for comp_idx in sel_components:
        final_mask[blobs == comp_idx] = 1
    '''
    final_mask = above_t1 * 0
    for comp_idx in range(1, n_components):
        comp_mask = blobs == comp_idx
        int_mask = comp_mask * above_t1
        if np.sum(int_mask):
            final_mask = final_mask + comp_mask
    final_mask[final_mask > 0] = 1
    '''
    return final_mask
