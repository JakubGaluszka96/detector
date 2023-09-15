import os
import json
from depthai_sdk import OakCamera

HOME = os.getcwd()

with OakCamera(usb_speed="usb3") as oak:
    color = oak.create_camera(source='color')
    nn = oak.create_nn(f"{HOME}/result/best_openvino_2022.1_5shave.blob",
    color, nn_type='yolo', spatial=True, tracker=True)
    #oak.visualize(nn, fps=True, scale=2/3)
    file = open(f"{HOME}/result/best.json")
    conf = json.load(file)
    nn.config_yolo_from_metadata(conf['nn_config']['NN_specific_metadata'])
    oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)