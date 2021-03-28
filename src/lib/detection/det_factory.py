
from .yolov3 import Darknet
from torchvision import transforms

class DetectorFactory:
    def create_detector(self, model_name):
        self.detection_options = ["yolo"]
        model = None
        if model_name == "yolo":
            model = Darknet("../models/yolo_config.cfg")
            model.load_weights("../models/yolov3-wider_16000.pt")
            model.eval()
            input_size = (416,416)
            return model, input_size
        else:
            raise Exception("Detection model must be selected from possible options: {}".format(self.detection_options))