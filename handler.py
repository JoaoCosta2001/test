import json
import torch
from ts.torch_handler.image_classifier import ImageClassifier

class CustomImageClassifier(ImageClassifier):
    def initialize(self, context):
        super().initialize(context)
        # Load ImageNet labels
        with open("/home/model-server/imagenet_classes.txt") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def postprocess(self, data):
        results = []
        for prediction in data:
            # Convert index to class label
            results.append({self.labels[i]: float(prediction[i]) for i in prediction})
        return results

_service = CustomImageClassifier()

