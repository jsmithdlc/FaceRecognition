import sys
import os
import os.path as osp
sys.path.insert(0, "/".join(osp.dirname(os.path.abspath(__file__)).split("/")[:-1]))

import numpy as np
import pandas as pd
from lib.inception_resnet_v1 import InceptionModelV1, load_weights
from PIL import Image
import glob
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import tqdm

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

class Database(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, imgs_path):
        'Initialization'
        self.imgs_list = glob.glob(imgs_path + "/*.jpg")
        self.imgs_names = [file.split("/")[-1].split("_")[0] for file in self.imgs_list]
        self.transformer = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img = Image.open(self.imgs_list[index])
        x = self.transformer(self.resize(img))
        return x

    def resize(self, img):
        img_resized = img.resize((160,160))
        return img_resized


def main(args):
    model = InceptionModelV1()
    model = load_weights(model,"../../models/20180402-114759-vggface2.pt")
    model = model.eval()
    model.to(device)
    dataset = Database(args.imgs_path)
    img_names = dataset.imgs_names
    embeddings = []
    iterator = DataLoader(dataset, batch_size = 8)
    print("Converting images to embeddings ...")
    for batch in tqdm.tqdm(iterator):
        batch = batch.to(device)
        batch_embeddings = model(batch)
        embeddings.append(batch_embeddings.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings,axis=0)
    print("Storing images to disk")
    df = pd.DataFrame(embeddings, index = img_names)
    df.index.name = "person_name"
    df.to_csv(r'../../database/embeddings/embeddings.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings to anchor images from database')
    parser.add_argument('imgs_path', type=str,
                         help='path to image directory.')
    args = parser.parse_args()
    main(args)

