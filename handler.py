from ts.torch_handler.image_classifier import ImageClassifier
import torch
from torchvision import transforms
from PIL import Image
import io

class CustomImageClassifier(ImageClassifier):
    def __init__(self):
        super(CustomImageClassifier, self).__init__()

    def initialize(self, context):
        """This method is called when the model is loaded into the worker."""
        super().initialize(context)
        # You can add custom initialization code here if needed.
        # The model will be loaded from the .mar file into `self.model`.
        self.manifest = context.manifest
        self.model = self.model  # Model is already loaded from ImageClassifier base class

    def preprocess(self, data):
        """Preprocess the input data."""
        image = data[0].get("body")
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            image = Image.open(image)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def inference(self, data, *args, **kwargs):
        """Perform inference."""
        # Assuming model is loaded and set to evaluation mode
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        return output

    def postprocess(self, data):
        """Postprocess the output."""
        # Assuming the model output is a tensor of logits
        probabilities = torch.nn.functional.softmax(data[0], dim=1)
        top_prob, top_class = torch.topk(probabilities, 1)
        return top_class.squeeze().tolist()
