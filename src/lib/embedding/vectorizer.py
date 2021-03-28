import sys
import os
sys.path.insert(0, "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]))

from .vec_factory import VectorizerFactory
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
import numpy as np
from utils.utils import *

# Factory for creating new detectors
vectorizer_factory = VectorizerFactory()

class FaceVectorizer:
    def __init__(self, model_name):
        self.model, self.input_size = vectorizer_factory.create(model_name)
        self.pil2torch = transforms.ToTensor()

    def preprocess_bboxes(self, img_size, detections, bbox_scale):
        width, height= img_size
        bbox_heights = detections[:,3] - detections[:,1]
        bbox_widths = detections[:,2] - detections[:,0]
        y_delta = bbox_heights * (bbox_scale-1)/2
        x_delta = bbox_widths * (bbox_scale-1)/2
        detections[:,0] = np.clip(detections[:,0] - x_delta, 0, width)
        detections[:,1] = np.clip(detections[:,1] - y_delta, 0, height)
        detections[:,2] = np.clip(detections[:,2] + x_delta, 0, width)
        detections[:,3] = np.clip(detections[:,3] + y_delta, 0, height)
        return detections

    def find_embeddings(self, img, detections):
        detections = self.preprocess_bboxes(img.size, detections, 1.3)
        embeddings =  []
        for idx, bbox in enumerate(detections):
            face  = img.crop(bbox[:4])
            pad_face, _ = pad_img(face, self.input_size)
            net_input = self.pil2torch(pad_face).unsqueeze(dim=0)
            embedding = self.model(net_input).detach().numpy()
            embeddings.append(embedding)
        return embeddings




