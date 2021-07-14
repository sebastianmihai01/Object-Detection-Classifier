import cv2  # computer vision
import uuid  # name our images uniquely
import os  # for paths
import time
import image_labelling as il
"""
Multi-Class object detection model -> it will be able to detect multiple gestures
"""

labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
# ideally, 15-20 images when starting, way more later on
number_imgs = 5  # 5 images for each label => 5x4= 20 images

# Setup folder, paths
# => creates just the path (not a folder) "Tensorflow\\workspace\\images\\collectedimages"
curr = os.getcwd() # dynamic file path
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
print("Current os: " + os.name)

# Create the folder directory given the path above
if not os.path.exists(IMAGES_PATH): # If not existing
    # If OS is linux
    # !mkdir -p {IMAGES_PATH}
    if os.name == 'posix':
        os.mkdir(IMAGES_PATH)

    # If OS is windows
    # !mkdir -p {IMAGES_PATH}
    if os.name == 'nt':
        il.execute("mkdir "+IMAGES_PATH+"")
        print("success")

# Create a folder for reach class (label) at the end of the PATH
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)


# Webcam DEVICE NUMBER based on OS
# Windows=0, Linux=1, Mac=2
device = 0  # For windows

# Image capturing
for label in labels:
    cap = cv2.VideoCapture(device)  # Connect to our webcam
    print('Collecting images for {}'.format(label))  # Specify the label which is captured
    time.sleep(5)  # Time range for the person to get a good angle

    # Loop through the number of images we want to collect
    # number_imgs = 5 in our case
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        # get frames
        ret, frame = cap.read()
        # 1) create a new image and put that into this particular folder path (IMAGES_PATH)
        # 2) label it (put it into the folder 'label')
        # 3) rename it with: label + random number using uuid1 + jpg format
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        # 'writes' the actual image (render)
        cv2.imwrite(imgname, frame)
        # show on screen
        cv2.imshow('frame', frame)
        # wait 2 seconds
        time.sleep(2)

        # exit via 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
