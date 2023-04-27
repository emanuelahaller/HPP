# Highly Probable Positive Features (HPP) 

This repository contains the code associated to paper [Unsupervised Object Segmentation in Video by Efficient Selection of Highly Probable Positive Features](https://openaccess.thecvf.com/content_iccv_2017/html/Haller_Unsupervised_Object_Segmentation_ICCV_2017_paper.html) by Emanuela Haller and Marius Leordeanu. 

The paper was published in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.

## Method Overview 

We address an essential problem in computer vision, that of unsupervised foreground object segmentation in video, where a main object of interest in a video sequence should be automatically separated from its background. An efficient solution to this task would enable large-scale video interpretation at a high semantic level in the absence of the costly manual labeling. We propose an efficient unsupervised method for generating foreground object soft masks based on automatic selection and learning from highly probable positive features. We show that such features can be selected efficiently by taking into consideration the spatio-temporal appearance and motion consistency of the object in the video sequence. We also emphasize the role of the contrasting properties between the foreground object and its background. Our model is created over several stages: we start from pixel level analysis and move to descriptors that consider information over groups of pixels combined with efficient motion analysis. We also prove theoretical properties of our unsupervised learning method, which under some mild constraints is guaranteed to learn
the correct classifier even in the unsupervised case. We achieve competitive and even state of the art results on the challenging Youtube-Objects and SegTrack datasets, while being at least one order of magnitude faster than the competition. We believe that the strong performance of our method, along with its theoretical properties, constitute a solid step towards solving unsupervised discovery in video.

## If you intend to use our work please cite this project as:
```
@inproceedings{haller2017unsupervised,
  title={Unsupervised object segmentation in video by efficient selection of highly probable positive features},
  author={Haller, Emanuela and Leordeanu, Marius},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5085--5093},
  year={2017}
}
```






