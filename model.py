import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torchvision import models, transforms
from torchvision.models import MobileNetV2
from PIL import Image


class OpenEyesClassificator:
    def __init__(self):
        self.preprocess: Compose = Compose(
            [
                transforms.Resize((24, 24)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.device: str = "cpu"
        self.model_path: str = "best_cpu.pth"
        self.model: MobileNetV2 = self.load_model(self.model_path, self.device)

    @staticmethod
    def load_model(model_path, device) -> MobileNetV2:
        model = models.mobilenet_v2(weights=None)
        model.features[0][0] = nn.Conv2d(
            1,
            model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias,
        )
        model.classifier[1] = nn.Linear(model.last_channel, 1)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        model = model.to(device)
        return model

    def predict(self, inpIm: str) -> float:
        image = Image.open(inpIm).convert("L")
        image = self.preprocess(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            is_open_score = torch.sigmoid(output).item()
        return is_open_score


if __name__ == "__main__":
    model = OpenEyesClassificator()
    print(model.predict("000400.jpg"))
