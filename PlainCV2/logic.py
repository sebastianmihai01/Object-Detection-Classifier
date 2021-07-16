import cv2
import TensorflowAPI.collectAndLabel.image_labelling as il
import matplotlib.pyplot as plt

# il.execute("pip install opencv-python")

config_file = '..\\model\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = '..\\coco\\frozen_inference_graph.pb'

# load the tf (pretrained) model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# from the coco labels, we make a list of class names (labels) separated by a comma
classLabels = []
file_name = '..\\model\\Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# read an image

img = cv2.imread('..\\catdog.jpg')
# print(img) shows the array of pixels (more like matrix)

plt.imshow(img) # draw the picture in GBR
# we however need RGB
# plt.show() # show the picture
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()

# resize the image to fit the model (320x320)
model.setInputSize(320, 320)

# 255 = grey level (before white 256)
model.setInputScale(1.0/127.5) # 255/2 = 127.5

# because mobilenet takes means [-1, 1]
# model.setInputMean(127.5, 127.5, 127.5)
model.setInputMean(127.5)
# automatic perform for every pic
model.setInputSwapRB(True)

# ClassIndex = [[1],[3]] 1-person, 3-car
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255,0 ,0 ), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale,
                color = (0,255,0), thickness=3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


