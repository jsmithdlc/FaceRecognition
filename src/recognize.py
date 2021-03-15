from lib.inception_resnet_v1 import InceptionModelV1
from lib.yolov3 import Darknet
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
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
    return detections, (x_scale, y_scale)

"""
def post_process(image, scales, detections):
    x_scale, y_scale = scales
    for box in detections:
        bbox = box.detach().numpy()
"""




def show_detections(img, scale, detections):
    x_scale, y_scale = scale
    draw = ImageDraw.Draw(img)
    for bbox in detections:
        x1, x2 = bbox[0] * x_scale, bbox[2] * x_scale
        y1, y2 = bbox[1] * y_scale, bbox[3] * y_scale
        draw.rectangle([x1,y1,x2,y2], width = 4)
    img.save("../out_imgs/detection", "JPEG")





if __name__ == '__main__':
    img_path = '/home/javier/Ramblings/FaceRecognition/sample_imgs/cabros.jpg'
    detections, scale = detect_people(img_path)
    show_detections(Image.open(img_path), scale, detections)
    """
    bbox = out_det[0].detach().numpy()[0]
    bbox[0], bbox[2] = bbox[0] * x_scale, bbox[2] * x_scale
    bbox[1], bbox[3] = bbox[1] * y_scale, bbox[3] * y_scale

    draw = ImageDraw.Draw(original_img)
    draw.rectangle(bbox[:4], width = 4)
    original_img.save("../out_imgs/lore_detected","JPEG")
    """






