from lib.inception_resnet_v1 import InceptionModelV1
from lib.yolov3 import Darknet
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from utils.utils import *

#def compare_imgs(name, img):

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

embeddings_model = InceptionModelV1()
embeddings_model.load_weights("../models/20180402-114759-vggface2.pt")
embeddings_model.eval()
anchor_embeddings = pd.read_csv("../database/embeddings/embeddings.csv",index_col=0)
embeddings_input_size = (160, 160)

yolo_model = Darknet("../models/yolo_config.cfg")
yolo_model.load_weights("../models/yolov3-wider_16000.pt")
yolo_model.eval()
yolo_input_size = (416,416)
pil2torch = transforms.ToTensor()

def detect_people(img_path):
    img = Image.open(img_path)
    size = img.size
    x_scale = size[0]/yolo_input_size[0]
    y_scale = size[1]/yolo_input_size[1]
    net_img = img.resize(yolo_input_size)

    detections = yolo_model(pil2torch(net_img).unsqueeze(dim=0))
    detections = non_max_suppression(detections)[0].detach().numpy()
    rescaled_detections = []
    for bbox in detections:
        bbox[0], bbox[2] = bbox[0] * x_scale, bbox[2] * x_scale
        bbox[1], bbox[3] = bbox[1] * y_scale, bbox[3] * y_scale
    return detections


def post_process(detections, im_size, bbox_scale):
    width, height= im_size
    bbox_heights = detections[:,3] - detections[:,1]
    bbox_widths = detections[:,2] - detections[:,0]
    y_delta = bbox_heights * (bbox_scale-1)/2
    x_delta = bbox_widths * (bbox_scale-1)/2
    detections[:,0] = np.clip(detections[:,0] - x_delta, 0, width)
    detections[:,1] = np.clip(detections[:,1] - y_delta, 0, height)
    detections[:,2] = np.clip(detections[:,2] + x_delta, 0, width)
    detections[:,3] = np.clip(detections[:,3] + y_delta, 0, height)
    return detections

def show_detections(img, detections):
    img_drawable = img.copy()
    draw = ImageDraw.Draw(img_drawable)
    for bbox in detections:
        x1, x2 = bbox[0], bbox[2]
        y1, y2 = bbox[1], bbox[3]
        draw.rectangle([x1,y1,x2,y2], width = 4)
    img_drawable.save("../out_imgs/detection", "JPEG")


def show_recognitions(img, recognitions):
    img_drawable = img.copy()
    draw = ImageDraw.Draw(img_drawable)
    font = ImageFont.truetype("../fonts/arial.ttf", int(img.size[0]*0.04))
    for recognition in recognitions.values():
        identification = recognition[0]
        bbox = recognition[1]
        color =  '#00FF00'
        if identification == "":
            color = '#FF0000'
        x1, x2 = bbox[0], bbox[2]
        y1, y2 = bbox[1], bbox[3]
        draw.rectangle([x1,y1,x2,y2], width = 4, outline = color)
        draw.text([x1+10,y1],identification.capitalize(),font=font, fill=color)
    img_drawable.show()

def recognize_detections(img, detections):
    recognitions = {}
    for idx, bbox in enumerate(detections):
        recognitions[idx] = bbox
        img_crop = img.crop(bbox[:4])
        crop_vector = embeddings_model(pil2torch(img_crop.resize(embeddings_input_size)).unsqueeze(dim=0)).detach().numpy()
        person_identified = ""
        min_distance = 10
        for person_name in anchor_embeddings.index.unique():
            anchor_vector = np.mean(anchor_embeddings.loc[person_name],axis=0).to_numpy()
            distance = np.linalg.norm(crop_vector-anchor_vector)
            if  (distance < 0.6) and (distance < min_distance):
                person_identified = person_name
                min_distance = distance
        recognitions[idx] = [person_identified,bbox]
    return recognitions


if __name__ == '__main__':
    img_path = '/home/javier/Ramblings/FaceRecognition/sample_imgs/couple.jpg'
    img = Image.open(img_path)
    detections = detect_people(img_path)
    post_process(detections, img.size, 1.3)
    recognitions = recognize_detections(img, detections)
    show_recognitions(img, recognitions)






