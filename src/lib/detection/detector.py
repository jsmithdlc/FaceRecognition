import sys
import os
sys.path.insert(0, "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]))

from .det_factory import DetectorFactory
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
import numpy as np
from utils.utils import *

# Factory for creating new detectors
detector_factory = DetectorFactory()

class Detector:
    def __init__(self, model_name):
        self.detector, self.input_size = detector_factory.create_detector(model_name)
        self.pil2torch = transforms.ToTensor()

    def postprocess(self, detections, scale, pad):
        detections = non_max_suppression(detections)[0].detach().numpy()      
        for bbox in detections:
            bbox[0] = (bbox[0] - pad[0])/scale
            bbox[2] = (bbox[2] - pad[0])/scale
            bbox[1] = (bbox[1] - pad[1])/scale
            bbox[3] = (bbox[3] - pad[1])/scale
        return detections
        
    def detect_people(self, img_or_path):
        if isinstance(img_or_path,str):
            try:
                img = Image.open(img_or_path)
            except NameError:
                print("Input image path is not valid")
        else:
            img = img_or_path

        padded_img,(scale, img_pad) = pad_img(img, self.input_size)    
        det_input = self.pil2torch(padded_img).unsqueeze(dim=0)
        detections = self.detector(det_input)
        detections = self.postprocess(detections, scale, img_pad)
        return detections

