from lib.inception_resnet_v1 import InceptionModelV1, load_weights
from PIL import Image
from torchvision import transforms
import numpy as np

def load_image(img_path):
	im = Image.open(img_path)
	im = im.resize((160,160))
	im_t = transforms.ToTensor()(im)
	return im_t.unsqueeze(0)

if __name__ == '__main__':
	model = InceptionModelV1()
	model = load_weights(model,"/home/javier/Ramblings/FaceRecognition/models/20180402-114759-vggface2.pt")
	model = model.eval()
	img1 = load_image("/home/javier/Ramblings/FaceRecognition/database/javier1.jpg")
	img2 = load_image("/home/javier/Ramblings/FaceRecognition/database/lore1.jpg")
	e1 = model(img1).detach().numpy()
	e2 = model(img2).detach().numpy()
	print(e1.shape)
	print(np.linalg.norm(e1-e2))

