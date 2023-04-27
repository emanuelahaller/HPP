"""
Main hpp function
"""
import time
from hpp import video_pca
from hpp import patch_analysis
from hpp import motion_analysis


def get_video_soft_segs(frames, enable_step1, enable_step2, enable_step3,
                        enable_step4):
    """
    Get video soft segs

    [in] frames - list of video frames
    [in] enable_step1 - if first step is enabled
    [in] enable_step2 - if second step is enabled
    [in] enable_step3 - if third step is enabled
    [in] enable_step4 - if fourth step is enabled

    [out] soft_segs - computed soft segs
    """
    if enable_step1:
        step1_soft_segs, step1_rec_imgs, step1_rec_diffs = video_pca.video_pca(
            frames[:])
    
    if enable_step2:
        step2_soft_segs, step2_rec_soft_segs = video_pca.video_pca_soft_segs(
            frames[:], step1_soft_segs[:])
    else:
        step2_soft_segs = None 
        step2_rec_soft_segs = None 

    if enable_step3:
        step3_soft_segs = patch_analysis.get_patch_based_soft_seg(
            frames[:], step2_soft_segs[:])#, n_features)
    else:
        step3_soft_segs = None 
    
    if enable_step4:
        motion_soft_segs = motion_analysis.get_motion_estimation(
            frames[:], step3_soft_segs[:])
        motion_soft_segs = motion_analysis.get_blurred_motion_estimation(
            motion_soft_segs[:])
        soft_segs = motion_analysis.comb_appearance_and_motion_info(\
            step3_soft_segs[:],\
            motion_soft_segs[:])
    else:
        motion_soft_segs = None 
        soft_segs = None 
        
    return step1_soft_segs, step1_rec_imgs, step1_rec_diffs, step2_soft_segs, step2_rec_soft_segs, step3_soft_segs, motion_soft_segs, soft_segs
    