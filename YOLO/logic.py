# weights wget https://pjreddie.com/media/files/yolov3.weights\

import cv2
import TensorflowAPI.collectAndLabel.image_labelling as il
import numpy as np

"""
> Weight file: it’s the trained model, the core of the algorythm to detect the objects.
> Cfg file: it’s the configuration file, where there are all the settings of the algorythm.
> Name files: contains the name of the objects that the algorythm can detect.
"""
# get darknet
"""
git clone https://github.com/pjreddie/darknet
cd darknet
make
"""
il.execute("git clone https://github.com/pjreddie/darknet")
il.execute("cd darknet && make")

# get the weights
il.execute("wget https://pjreddie.com/media/files/yolov3.weights")

# get the cfg
il.execute("wget https://pjreddie.com/media/files/darknet53.conv.74")
il.execute("./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74")
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # network

# load classes from the coco file
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# get layer names
layer_names = net.getLayerNames()

# get output layers
# we need these to get the final result/ detection of the objects (objects desplayed on the screen)
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("catdog.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
# Conversion to a BLOB, we cannot pass the image as it is to the NN
# 0.00392 = scale factor
# base size : 416x416
# True = invert red with blue because openCV works with BGR format, not RGB
# Not cropping, we want to detect everything

""" >>>>>>> Blob has 3 images: each for every GBR channel (green, blue, red) => 3 different-coloured images in one blob

Keep in mind that we can’t use right away the full image on the network, but first we need it to convert it to blob. 
Blob it’s used to extract feature from the image and to resize them. YOLO accepts three sizes:

> 320×320 it’s small so less accuracy but better speed
> 609×609 it’s bigger so high accuracy and slow speed
> 416×416 it’s in the middle and you get a bit of both.

"""
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# pass the blob to the NN
net.setInput(blob)

# forward to the architecture
outs = net.forward(output_layers)

""" Now, we have all the information in order to extract the objects that we want """



# Showing informations on the screen
class_ids = []

"""
We then loop trough the outs array, we calculate the confidence and we choose a confidence threshold.

On line 32 we set a threshold confidence of 0.5, if it’s greater we consider the object correctly detected, 
otherwise we skip it.
The threshold goes from 0 to 1. The closer to 1 the greater is the accuracy of the detection, while the closer to 0
 the less is the accuracy but also it’s greater the number of the objects detected.

"""
# Confidence = how sure the algorithm is that the detection was correct
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


"""
When we perform the detection, it happens that we have more boxes for the same object, 
so we should use another function to remove this “noise”.
It’s called Non maximum suppression.
"""

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

"""
We finally extract all the information and show them on the screen.

Box: contain the coordinates of the rectangle surrounding the object detected.
Label: it’s the name of the object detected
Confidence: the confidence about the detection from 0 to 1.
"""

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()