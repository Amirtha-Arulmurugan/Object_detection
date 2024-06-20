#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

# Load YOLOv3 config and weights
net = cv2.dnn.readNet('C:/Users/Admin/yolov3.cfg','C:/Users/Admin/yolov3.weights')

# Load COCO object names
with open('C:/Users/Admin/coco.names', "r") as f:
    classes = f.read().strip().split("\n")

# Load an image
image = cv2.imread('C:/Users/Admin/horse.jpg')

# Get image dimensions and create a blob from it
height, width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the network and make predictions
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Initialize lists to store class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Process the output and filter out weak detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # You can adjust the confidence threshold
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
            x, y = center_x - w // 2, center_y - h // 2

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Loop over the detected objects and draw bounding boxes and labels
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # You can change the color
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with detected objects
cv2.imshow("YOLOv3 Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

