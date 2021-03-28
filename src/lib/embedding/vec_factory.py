from .inception_resnet_v1 import InceptionModelV1

class VectorizerFactory:
    def create(self, model_name):
        self.vectorizer_options = ["facenet"]
        model = None
        if model_name == "facenet":
            model = InceptionModelV1()
            model.load_weights("../models/20180402-114759-vggface2.pt")
            model.eval()
            input_size = (160,160)
            return model, input_size
        else:
            raise Exception("Face Vectorization model must be selected from possible options: {}".format(self.vectorizer_options))