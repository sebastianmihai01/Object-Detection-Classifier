"""
Current model used: SSD (single shot detection) MobileNet V2 320x320 (compressing our image to 320x320)
_____________________

> Model is trained on COCO dataset
> The output & how good the model is performing can be found on the website below

Advantages:
_____________________
1) We need to trade off SPEED for ACCURACY
   - eg: for raspberry pi/ telephone => speed (because they dont have GPU acceleration)
   - hard to have both, might as well choose to rely on a single one

2) Preprocessing and Postporcessing is done automatically
   - meaning: compression - for accuracy (to 320x320) and decompression (going back to the original resolution)

3) Augmentation


read more on TF model ZOO: https://github.com/tensorflow/models/tree/master/research/object_detection/models
"""