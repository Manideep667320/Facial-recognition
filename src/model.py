# src/model.py
import torch
import torch.nn as nn
import torchvision

class ResNetEmbedding(nn.Module):
    """
    ResNet backbone -> embedding -> classifier
    - backbone: torchvision resnet
    - embedding: Linear(in_feats, emb_size) + BN + ReLU
    - classifier: Linear(emb_size, num_classes)
    """
    def __init__(self, backbone='resnet50', emb_size=512, num_classes=31, pretrained=True):
        super().__init__()
        # instantiate backbone
        if not hasattr(torchvision.models, backbone):
            raise ValueError(f"Backbone {backbone} not found in torchvision.models")
        model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        in_feats = model.fc.in_features
        # remove fc and pooling left as-is (avgpool remains)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # all layers up to avgpool
        self.embedding = nn.Sequential(
            nn.Linear(in_feats, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x, return_embedding=False):
        # x -> backbone -> flatten -> embedding -> logits
        x = self.backbone(x)            # shape: (B, C, 1, 1)
        x = x.view(x.size(0), -1)       # shape: (B, C)
        emb = self.embedding(x)         # (B, emb_size)
        logits = self.classifier(emb)   # (B, num_classes)
        if return_embedding:
            return logits, emb
        return logits
