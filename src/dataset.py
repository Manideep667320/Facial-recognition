# src/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

def get_train_transform(img_size=224, re_prob=0.2):
    # Using torchvision transforms; you may choose albumentations for more variety.
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        # RandomErasing included as a transform; torchvision >=0.9 required
        transforms.RandomErasing(p=re_prob, scale=(0.02,0.33), ratio=(0.3,3.3), value=0)
    ])

def get_valid_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

class FolderDataset(Dataset):
    """
    Expects `root` directory with subfolders for each class.
    sample format:
      root/
        classA/
          img1.jpg
          img2.jpg
        classB/
          ...
    Returns: (image_tensor, label_idx, image_path)
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            p = os.path.join(root, c)
            for fname in sorted(os.listdir(p)):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append((os.path.join(p, fname), self.class_to_idx[c]))
        self.classes = classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label, path
