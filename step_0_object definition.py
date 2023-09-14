# first, import all necessary modules
from pathlib import Path

import blobconverter
import cv2
import depthai
import numpy as np

background = None
MAX_FRAMES = 1000
THRESH = 60
ASSIGN_VALUE = 255
SIZE = (384,384)

def make_yolo_label(cls, x, y, w, h, size):
    center_x = (x+w/2)/size[0]
    center_y = (y+h/2)/size[1]
    yolo_w=w/size[0]
    yolo_h=h/size[1]
    print(cls, center_x, center_y, yolo_w, yolo_h)
    return None

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(SIZE)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)


# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)


# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None

    # Main host-side application loop
    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()

        if frame is not None:
            cv2.imshow("preview", frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
            cv2.imshow("greyscale", frame_gray)

        if cv2.waitKey(3) == ord('b'):
            # Train background with first frame
            background = frame_gray
            cv2.imshow('Background', background)

        if background is not None:
            diff = cv2.absdiff(background, frame_gray)
            # Mask thresholding
            ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, cv2.THRESH_BINARY)
            (x, y, w, h) = cv2.boundingRect(motion_mask)
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)
            cv2.imshow("motion", motion_mask)
           
            # at any time, you can press "w" to capture an image 
        if cv2.waitKey(2) == ord('w'):
            value = input("Wybierz klasę obiektu:\n")
            make_yolo_label(value, x, y, w, h, SIZE)

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break



     