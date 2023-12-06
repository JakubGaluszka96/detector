# Detector

Scrip program created to obtain training data and to train YOLO V8 model. Accuaracy of the model has been checked using 5 types of samples which are set of lines with different thickness. Images available to print in "samples" folder. 

# Setup

It is obligatory to have Luxonis camera connected to your computer. If not you can only validate the model using web camera.

Python is required 

> sudo apt install python3

All dependecies are able to run in virtual enviroment

> python3 -m venv venv

> source venv/bin/activate

Install requirements:

> sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash

> pip install opencv-python

> pip install depthai

> pip install ultralytics

> pip install depthai_sdk

## Model development is devided on 3 steps:

### Step 0 Object definition

Run sctipt step 0
Inside venv:

> ./venv/bin/python "./step_0_object definition.py"

1. Using "b" button take the picture of the background
2. Now insert object into camera visible area
3. Frame around the object should occur
4. Press "w" to capture image
5. Type number 0-4 and press Enter. Each number is associated with one class of captured object
6. Repeat points 2-5 in order to obtain representative amount of data

### Step 1 Training

Run:

> ./venv/bin/python "./step_1_training.py"

Models in .pt format will be stored in ./runs/detect/train

### Step 2 Conversion to blob

Go to: 

> https://tools.luxonis.com/

Choose file from ./runs/detect/train/weights/best.pt

Make setup as in the image "Yolov8 conversion"

Converted model will be downloaded automaticly. Extract content into:

./result

### Step 3 Live check

Run:

./venv/bin/python "./step_3_live_check_oak_camera.py"

# For those who don't have Oak camera it is plausible to check model on web camera

./venv/bin/python "./step_3_live_check_web_camera.py"




