
fabric_datasett - v4 2025-11-30 8:01pm
==============================

This dataset was exported via roboflow.com on November 30, 2025 at 12:02 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 288 images.
Objects-XCKZ are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 524x524 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random rotation of between -15 and +15 degrees
* Random brigthness adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 3.03 percent of pixels


