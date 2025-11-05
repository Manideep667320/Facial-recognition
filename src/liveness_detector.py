import cv2
import numpy as np
import torch
from model import ResNetEmbedding
from torchvision import transforms
from PIL import Image
import os
import warnings

class LivenessDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Try to load the model, but don't fail if it's not available
        if model_path and os.path.exists(model_path):
            try:
                self.model = self._load_model(model_path)
                print(f"Successfully loaded liveness detection model from {model_path}")
            except Exception as e:
                warnings.warn(f"Failed to load liveness detection model: {str(e)}. Continuing without liveness detection.")
        else:
            warnings.warn(f"Liveness model not found at {model_path}. Continuing without liveness detection.")
    
    def _load_model(self, model_path):
        """Load the pre-trained liveness detection model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = ResNetEmbedding(backbone='resnet18', emb_size=128, num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
        
    def is_available(self):
        """Check if the liveness detector is available."""
        return self.model is not None
        
    def detect(self, frame, face_box):
        """Detect liveness for a single face"""
        if self.model is None:
            # If model is not available, return True (real) with high confidence
            return True
            
        try:
            x, y, w, h = face_box
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(face_img)
            face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(face_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                live_prob = prob[0][1].item()  # Probability of being a real face
                
            return live_prob > 0.85  # Threshold for liveness
            
        except Exception as e:
            warnings.warn(f"Error in liveness detection: {str(e)}")
            return True  # Default to real if there's an error
