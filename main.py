import cv2
import os
import shutil
import sys
import numpy as np
import hpp.main_hpp

WORKING_SIZE = 300 

def get_images(video_path):
    frames = []
    img_paths = os.listdir(video_path)
    img_paths.sort()
    for img_path in img_paths:
        frame = cv2.imread(os.path.join(video_path, img_path))
        height = frame.shape[0]
        width = frame.shape[1]
        frame = cv2.resize(  #pylint:disable=no-member
            src=frame,
            dsize=(WORKING_SIZE, WORKING_SIZE),
            interpolation=cv2.INTER_CUBIC)  #pylint:disable=no-member
        frames.append(frame)
    return frames

def process_movies(VIDEOS_PATH, MAIN_OUT_PATH, ENABLE_STEP_1, ENABLE_STEP_2, ENABLE_STEP_3, ENABLE_STEP_4):
    videos = os.listdir(VIDEOS_PATH)
    videos.sort()
    for video_name in videos:
        vid_out_path = os.path.join(MAIN_OUT_PATH, video_name)
        os.makedirs(vid_out_path, exist_ok=True)
        frames = get_images(os.path.join(VIDEOS_PATH, video_name))

        step1_soft_segs, step1_rec_imgs, step1_rec_diffs, step2_soft_segs, step2_rec_soft_segs, step3_soft_segs, step4_motion_soft_segs, step4_soft_segs = hpp.main_hpp.get_video_soft_segs(
            frames[:], ENABLE_STEP_1, ENABLE_STEP_2, ENABLE_STEP_3, ENABLE_STEP_4)

    
        if ENABLE_STEP_1 == 1 and (ENABLE_STEP_2==0 and ENABLE_STEP_3==0 and ENABLE_STEP_4==0):
            empty_frame = np.zeros((300, 300, 3))
            for i in range(len(frames)):
                img = frames[i]
                step1_rec = step1_rec_imgs[i]
                step1_rec[step1_rec < 0] = 0
                step1_rec[step1_rec > 1] = 1
                step1_rec = step1_rec * 255

                step1_rec_diff = step1_rec_diffs[i]
                step1_rec_diff = step1_rec_diff[:, :, None]
                step1_rec_diff = np.repeat(step1_rec_diff, 3, 2)
                step1_rec_diff = step1_rec_diff * 255

                step1_soft_seg = step1_soft_segs[i]
                step1_soft_seg = step1_soft_seg[:, :, None]
                step1_soft_seg = np.repeat(step1_soft_seg, 3, 2)
                step1_soft_seg = step1_soft_seg * 255
                line0 = np.concatenate((img, empty_frame, empty_frame), 1)
                line1 = np.concatenate((step1_rec, step1_rec_diff, step1_soft_seg),
                                   1)
                img = np.concatenate((line0, line1))
                cv2.imwrite(os.path.join(vid_out_path, '%05d.png' % i),
                            np.uint8(img))
        else:
            if ENABLE_STEP_4:
                to_display_soft_segs = step4_soft_segs
            elif ENABLE_STEP_3:
                to_display_soft_segs = step3_soft_segs
            elif ENABLE_STEP_2:
                to_display_soft_segs = step2_soft_segs
            elif ENABLE_STEP_1:
                to_display_soft_segs = step1_soft_segs
            for i in range(len(frames)):
                img = frames[i]
                soft_seg = to_display_soft_segs[i]
                soft_seg = soft_seg[:, :, None]
                soft_seg = np.repeat(soft_seg, 3, 2)
                soft_seg = soft_seg * 255

                img = np.concatenate((img, soft_seg))
                cv2.imwrite(os.path.join(vid_out_path, '%05d.png' % i),
                            np.uint8(img))


if __name__=='__main__':
    videos_path = '/root/code/hpp2023_tests/videos'
    main_out_path = '/root/code/hpp2023_tests/videos_results'

    ENABLE_STEP_1 = 1
    ENABLE_STEP_2 = 1
    ENABLE_STEP_3 = 1
    ENABLE_STEP_4 = 1

    os.makedirs(main_out_path, exist_ok = True)
    process_movies(videos_path, main_out_path, ENABLE_STEP_1, ENABLE_STEP_2, ENABLE_STEP_3, ENABLE_STEP_4)
