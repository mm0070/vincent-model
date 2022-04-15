
vincent - v4 vincent_no_bed_augmentations_v2
==============================

This dataset was exported via roboflow.ai on April 15, 2022 at 6:53 PM GMT

It includes 285 images.
Vincents are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -19 and +19 degrees
* Random Gaussian blur of between 0 and 0.75 pixels
* Salt and pepper noise was applied to 5 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Randomly crop between 0 and 20 percent of the bounding box
* Random Gaussian blur of between 0 and 6.25 pixels


