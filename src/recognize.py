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



if __name__ == '__main__':
    embeddings_model = InceptionModelV1()
    embeddings_model.load_weights("../models/20180402-114759-vggface2.pt")
    embeddings_model.eval()
    anchor_embeddings = pd.read_csv("../database/embeddings/embeddings.csv",index_col=0)

    yolo_model = Darknet("../models/yolo_config.cfg")
    yolo_model.load_weights("../models/yolov3-wider_16000.pt")
    yolo_model.eval()

    original_img = Image.open('/home/javier/Ramblings/FaceRecognition/sample_imgs/lore_boat.jpg')
    original_size = original_img.size
    net_size = (416,416)
    x_scale = original_size[0]/416
    y_scale = original_size[1]/416

    net_img = original_img.resize(net_size)


    out_det = yolo_model(transforms.ToTensor()(net_img).unsqueeze(dim=0))
    out_det = non_max_suppression(out_det)
    bbox = out_det[0].detach().numpy()[0]
    bbox[0], bbox[2] = bbox[0] * x_scale, bbox[2] * x_scale
    bbox[1], bbox[3] = bbox[1] * y_scale, bbox[3] * y_scale

    draw = ImageDraw.Draw(original_img)
    draw.rectangle(bbox[:4], width = 4)
    original_img.save("../out_imgs/lore_detected","JPEG")






