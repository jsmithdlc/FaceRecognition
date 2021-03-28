from lib.embedding.vectorizer import FaceVectorizer
from lib.detection.detector import Detector

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

anchor_embeddings = pd.read_csv("../database/embeddings/embeddings.csv",index_col=0)
embeddings_input_size = (160, 160)


def show_detections(img, detections):
    img_drawable = img.copy()
    draw = ImageDraw.Draw(img_drawable)
    for bbox in detections:
        x1, x2 = bbox[0], bbox[2]
        y1, y2 = bbox[1], bbox[3]
        draw.rectangle([x1,y1,x2,y2], width = 4)
    img_drawable.show()


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

def recognize_detections(img, detections, embeddings):
    recognitions = {}
    for idx, bbox in enumerate(detections):
        recognitions[idx] = bbox
        face_vector = embeddings[idx]
        person_identified = ""
        min_distance = 10
        for person_name in anchor_embeddings.index.unique():
            anchor_vector = np.mean(anchor_embeddings.loc[person_name],axis=0).to_numpy()
            distance = np.linalg.norm(face_vector-anchor_vector)
            if  (distance < 0.45) and (distance < min_distance):
                person_identified = person_name
                min_distance = distance
        recognitions[idx] = [person_identified,bbox]
    return recognitions


if __name__ == '__main__':
    img_path = '/home/javier/Ramblings/FaceRecognition/sample_imgs/lore_boat.jpg'
    img = Image.open(img_path)
    det = Detector("yolo")
    faceVec = FaceVectorizer("facenet")

    detections = det.detect_people(img)
    embeddings = faceVec.find_embeddings(img, detections)

    recognitions = recognize_detections(img, detections, embeddings)
    show_recognitions(img, recognitions)






