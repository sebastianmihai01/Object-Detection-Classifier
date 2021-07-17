import cv2

cap = cv2.VideoCapture("highway.mp4")

# Object detection from Stable camera
"""
As you can see in the example code we also used the createBackgroundSubtractorMOG2 function which 
Returns the “background ratio” parameter of the algorithm and then create the mask.


Mask = BLACK AND WHITE image, with contours on silhouettes
"""
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    # 1. Object Detection
    mask = object_detector.apply(frame)

    """
    
    As you can see, however, there is a lot of noise in the image. So let’s improve the extraction by
     removing all the smaller elements and focus our attention on objects that are larger than a certain area.
    
    Remove noise (white dots, things that are not important and disturb our black&white image
    """

    """
    Drawing the contours with OpenCV’s cv2.drawContours function we obtain this result. 
    You won’t need to use this function, consider it as a debug of a first result
    """
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
       # Calculate area and remove small elements
       area = cv2.contourArea(cnt)
       if area > 100:
           pass


    """
    We define a Region of interest
    For the purpose of this tutorial, it is not important to analyze the entire window. 
    We are only interested in counting all the vehicles that pass at a certain point, for this reason, 
    we must define a region of interest ROI and apply the mask only in this area.
    
    
    RESULT: green lines only on the objects in a particular area
    """
    while True:
        ret, frame = cap.read()
        height, width, _ = frame.shape
        # Extract Region of interest
        roi = frame[340: 720, 500: 800]
        # 1. Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 100:
                cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)


"""
The function cv2.createBackgroundSubtractorMOG2 was added at the beginning without defining parameters, 
now let’s see how to further improve our result. history is the first parameter, in this case, 
it is set to 100 because the camera is fixed. var Threshold instead is 40 because the lower the value the greater the possibility of making false positives. In this case, we are only interested in the larger objects.
"""

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


"""
Draw the box around the object
Before proceeding with the rectangle we do a further cleaning of the image. To do this, the threshold
 function comes in handy. Starting from our mask we tell it that we want to show only the white or black values
  so by writing “254, 255” only the values between 254 and 255 will be considered.
"""

_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

# ---------------------------------------------------------------

""" Tracking functions """

from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

# Once the object has been created, we must therefore take each position of the bounding box and
# insert them in a single array.

detections.append([x, y, w, h])


"""
Associate unique ID to the object
Let’s now pass our array with positions to tracker.update(). 
We will again get an array with the potions but in addition, a unique id will be assigned for each object.

As you can see from the code we can analyze everything with a for a loop. 
At this point we just have to draw the rectangle and show the vehicle ID.
"""

# 2. Object Tracking
boxes_ids = tracker.update(detections)
for box_id in boxes_ids:
    x, y, w, h, id = box_id
    cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)