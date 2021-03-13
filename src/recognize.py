from lib.inception_resnet_v1 import InceptionModelV1, load_weights
from lib.yolov3 import Darknet
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

#def compare_imgs(name, img):

#def detect_people(img):



if __name__ == '__main__':
    embeddings_model = InceptionModelV1()
    embeddings_model = load_weights(embeddings_model,"../models/20180402-114759-vggface2.pt")
    embeddings_model = model.eval()
    anchor_embeddings = pd.read_csv("../database/embeddings/embeddings.csv",index_col=0)

    yolo_model = Darknet("../models/yolo_config.cfg")


