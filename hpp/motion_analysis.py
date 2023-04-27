"""
Motion analysis functions
"""
import sys
import cv2
import numpy as np
from hpp import common


def aux_get_xy_coords(n_rows, n_cols):
    """
    Aux function of get_motion_estimation
    Get x & y grids

    [in] n_rows - nr of rows in considered image
    [in] n_cols - nr of columns in considered image

    [out] x_coords - x coords
    [out] y_coords - y coords
    """
    x_coords = np.arange(0, n_cols)
    x_coords = x_coords[np.newaxis, :]
    x_coords = np.repeat(x_coords, n_cols, axis=0)

    y_coords = np.arange(0, n_rows)
    y_coords = y_coords[:, np.newaxis]
    y_coords = np.repeat(y_coords, n_rows, axis=1)

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    x_coords = x_coords[:, np.newaxis]
    y_coords = y_coords[:, np.newaxis]

    return x_coords, y_coords


def aux_get_img_derivatives(img_0, img_1, img_2):
    """
    Aux function for get_motion_estimation
    Get image derivatives

    [in] img_0 - previous frame
    [in] img_1 - current frame
    [in] img_2 - next frame

    [out] img_dx - derivative w.r.t to x
    [out] img_dy - derivative w.r.t to y
    [out] img_dt - derivative w.r.t to t
    """
    #nonlocal xfilter
    #nonlocal yfilter

    xfilter = np.array([-1, 0, 1])
    xfilter = xfilter[np.newaxis, :]
    yfilter = np.array([-1, 0, 1])
    yfilter = yfilter[:, np.newaxis]

    img_0 = common.normalize_tensor(img_0)
    img_1 = common.normalize_tensor(img_1)
    img_2 = common.normalize_tensor(img_2)

    img_dx = cv2.filter2D(img_1, ddepth=-1, kernel=xfilter)  #pylint:disable=no-member
    img_dy = cv2.filter2D(img_1, ddepth=-1, kernel=yfilter)  #pylint:disable=no-member
    img_dt = img_2 - img_0

    return img_dx, img_dy, img_dt


def aux_get_bg_data(soft_seg, img_dx, img_dy, img_dt):
    """
    Aux function for get_motion_estimation
    Extract background data

    [in] soft_seg - soft segmentation of the considered frame
    [in] img_dx - image derivative w.r.t. x
    [in] img_dy - image derivative w.r.t. y
    [in] img_dt - image derivative w.r.t. t

    [out] ix_bg - image derivative w.r.t x, in background pos
    [out] iy_bg - image derivative w.r.t y, in background pos
    [out] it_bg - image derivative w.r.t t, in background pos
    [out] xs_bg - x coords in background pos
    [out] ys_bg - y coords in background pos
    """
    background = 1 - soft_seg
    background[background < 0.8] = 0
    background[background >= 0.8] = 1

    background_pos = background > 0

    ix_bg = img_dx[background_pos]
    iy_bg = img_dy[background_pos]
    it_bg = img_dt[background_pos]

    pos = np.nonzero(background_pos)
    ys_bg = pos[0]
    xs_bg = pos[1]

    ix_bg = ix_bg[:, np.newaxis]
    iy_bg = iy_bg[:, np.newaxis]
    it_bg = it_bg[:, np.newaxis]
    xs_bg = xs_bg[:, np.newaxis]
    ys_bg = ys_bg[:, np.newaxis]

    return ix_bg, iy_bg, it_bg, xs_bg, ys_bg


def aux_get_grayscale_imgs(frames):
    """
    Aux function for get_motion_estimation
    Transform list of frames in grayscale frames

    [in] frames - list of BGR frames

    [out] frames - list of grayscale frames
    """
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #pylint:disable=no-member
        frames[idx] = frame

    return frames


def get_motion_estimation(frames, soft_segs):
    """
    Compute motion based foreground object soft segmentations

    [in] frames - list of frames
    [in] soft_segs - list of appearance soft segs

    [out] list of motion based soft segmentations
    """
    motion_soft_segs = []

    frames = aux_get_grayscale_imgs(frames)

    x_coords, y_coords = aux_get_xy_coords(frames[0].shape[0],
                                           frames[0].shape[1])

    for idx in np.arange(1, len(frames) - 1):
        img_dx, img_dy, img_dt = aux_get_img_derivatives(
            frames[idx - 1], frames[idx], frames[idx + 1])

        ix_bg, iy_bg, it_bg, xs_bg, ys_bg = aux_get_bg_data(
            soft_segs[idx], img_dx, img_dy, img_dt)

        data = np.concatenate((ix_bg, iy_bg, xs_bg * ix_bg, ys_bg * iy_bg,
                               xs_bg * iy_bg, ys_bg * ix_bg),
                              axis=1)

        #print('%d'%(idx))
        if np.min(data) == 0 and np.max(data) == 0:
            motion_soft_segs.append(
                np.zeros((frames[0].shape[0], frames[0].shape[1])))
        else:
            weights = np.dot(
                np.dot(np.linalg.inv(np.dot(data.T, data)), data.T), it_bg)

            img_dx = img_dx.flatten()
            img_dy = img_dy.flatten()
            img_dt = img_dt.flatten()

            img_dx = img_dx[:, np.newaxis]
            img_dy = img_dy[:, np.newaxis]
            img_dt = img_dt[:, np.newaxis]

            data = np.concatenate(
                (img_dx, img_dy, x_coords * img_dx, y_coords * img_dy,
                 x_coords * img_dy, y_coords * img_dx),
                axis=1)

            rasp = np.abs(np.dot(data, weights) - img_dt)
            rasp = np.reshape(rasp, (frames[0].shape[0], frames[0].shape[1]))

            motion_soft_segs.append(common.normalize_tensor(rasp))

    motion_soft_segs = [motion_soft_segs[0]] + motion_soft_segs
    motion_soft_segs.append(motion_soft_segs[len(motion_soft_segs) - 1])

    return motion_soft_segs


def aux_approximate_sigma(motion_soft_segs):
    """
    Aux function fro get_blurred_motion_estimation
    Approximate parameters of gaussian used for blurring the motion segs

    [in] motion_soft_seg - considered motion soft segmentations

    [out] sigma - computed sigma
    [out] dimx - x dimension for gaussian kernel
    [out] dimy - y dimension for gaussian kernel
    """
    cnt = 0
    std_x = []
    std_y = []
    for seg in motion_soft_segs:
        seg = common.normalize_tensor(seg)
        pos = np.nonzero(seg > 0.75)
        y_coords = pos[0]
        x_coords = pos[1]

        if len(x_coords) > 0:
            std_x.append(np.std(x_coords))
            std_y.append(np.std(y_coords))
            cnt = cnt + 1

    mean_std_x = np.mean(std_x)
    mean_std_y = np.mean(std_y)

    dim_x = np.int32(np.round(mean_std_x * 4))
    dim_y = np.int32(np.round(mean_std_y * 4))

    sigma = np.int32(2 * np.round(np.mean([mean_std_x, mean_std_y])))

    return sigma, dim_x, dim_y


def get_blurred_motion_estimation(motion_soft_segs):
    """
    Get blurred motion soft segs

    [in] motion_soft_segs - estimated motion soft segmentations
    [out] blurred_motion_soft_segs - blurred motion soft segmentations
    """
    n_frames = len(motion_soft_segs)
    sigma, dim_x, dim_y = aux_approximate_sigma(motion_soft_segs)

    if sigma == 0:
        blurred_motion_soft_segs = []
        for i in range(n_frames):
            blurred_motion_soft_segs.append(np.ones(motion_soft_segs[0].shape))
    else:
        gauss_kernel = common.gauss_kernel(shape=(dim_y, dim_x), sigma=sigma)
        blurred_motion_soft_segs = []
        for i in range(n_frames):
            seg = motion_soft_segs[i]
            seg = cv2.filter2D(seg, ddepth=-1, kernel=gauss_kernel)  #pylint:disable=no-member
            seg = common.normalize_tensor(seg)
            blurred_motion_soft_segs.append(seg)

    return blurred_motion_soft_segs


def comb_appearance_and_motion_info(soft_segs, motion_soft_segs):
    """
    Combine motion and appearance soft segmentations

    [in] soft_segs - appearance soft segmentations
    [in] motion_soft_segs - motion soft segmentations

    [out] final_soft_segs - final soft segmentations
    """
    min_val = sys.float_info.max
    max_val = sys.float_info.min
    final_soft_segs = []
    for idx, soft_seg in enumerate(soft_segs):
        soft_seg = soft_seg * motion_soft_segs[idx]
        min_val = min(min_val, np.min(soft_seg))
        max_val = max(max_val, np.max(soft_seg))
        final_soft_segs.append(soft_seg)

    for idx, soft_seg in enumerate(final_soft_segs):
        soft_seg = (soft_seg - min_val) / (max_val - min_val +
                                           sys.float_info.epsilon)
        final_soft_segs[idx] = soft_seg

    return final_soft_segs
