import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .alexnet import AlexNet
from .lenet import LeNet
from .resnet import ResNet18


def build_model(config):
    "Model builder."

    model_type = config.MODEL.NAME

    if model_type == 'lenet':
        model = LeNet(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'alexnet':
        model = AlexNet(num_classes=config.MODEL.NUM_CLASSES, dropout=config.MODEL.DROP_RATE)
    elif model_type == 'resnet18':
        model = ResNet18(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'resnet50_pretrained':
        # ImageNet-pretrained ResNet-50; swap the 1000-class head for our classes
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    return model
