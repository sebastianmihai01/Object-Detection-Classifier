import cv2  # computer vision
import uuid  # name our images uniquely
import os  # for paths
import time

"""
Multi-Class object detection model -> it will be able to detect multiple gestures
"""

labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
# ideally, 15-20 images when starting, way more later on
number_imgs = 5  # 5 images for each label => 5x4= 20 images

# Setup folder, paths
# => creates a folder with path "Tensorflow\\workspace\\images\\collectedimages"
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')


