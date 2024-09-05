import torch
import torchvision.models as models
from pathlib import Path

# Baixe o modelo resnet-18 pré-treinado
resnet18_model = models.resnet18(pretrained=True)
resnet18_model.eval()

# Baixe o modelo densenet-121 pré-treinado
densenet_model = models.densenet121(pretrained=True)
densenet_model.eval()

# Salva o modelo resnet-18 como um arquivo .pt
resnet18_model_path = Path("resnet-18.pt")
torch.jit.save(torch.jit.script(resnet18_model), resnet18_model_path)

# Salva o modelo densenet-121 como um arquivo .pt
densenet_model_path = Path("densenet-121.pt")
torch.jit.save(torch.jit.script(densenet_model), densenet_model_path)
