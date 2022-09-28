from transformers import ViTModel, ViTFeatureExtractor
from transformers import ViTForImageClassification
import torch.nn as nn

class My_Vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.vit.config.hidden_size, 1000)
    def forward(self, img):
        output = self.vit(img)
        return self.classifier(output.last_hidden_state[:, 0, :])

