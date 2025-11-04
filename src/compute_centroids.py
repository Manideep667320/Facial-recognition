# src/compute_centroids.py
"""
Compute L2-normalized centroids (per-class) using a saved checkpoint.

Usage:
  python src/compute_centroids.py --weights checkpoints/best.pth --train_dir data/train --out checkpoints/centroids.pth
Output:
  torch.save({'centroids': np.array (C,D), 'classes': [class_names...]}, out)
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import ResNetEmbedding

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to model checkpoint')
    p.add_argument('--train_dir', default='data/train', help='Directory containing training images organized in subfolders by class')
    p.add_argument('--out', default='checkpoints/centroids.pth', help='Output path for centroids')
    p.add_argument('--img_size', type=int, default=224, help='Input image size')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                  help='Device to use for computation')
    return p.parse_args()

def load_model(weights_path, device):
    """Load model from checkpoint with proper handling of architecture mismatches"""
    # First, get the number of classes from class_index.json
    with open('dataset/meta/class_index.json', 'r') as f:
        class_index = json.load(f)
    num_classes = len(class_index)
    
    # Print model architecture info
    print(f"\nLoading model with {num_classes} classes")
    
    # Initialize model with default parameters first
    model = ResNetEmbedding(
        backbone='resnet50',
        emb_size=512,  # Default, will be updated if different in checkpoint
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(weights_path, map_location='cpu')
    model_state = ckpt.get('model_state', ckpt)  # Handle both full checkpoint and state_dict
    
    # Print model state keys for debugging
    print("\nModel state keys:")
    for key in list(model_state.keys())[:5]:
        print(f"  {key}: {model_state[key].shape if hasattr(model_state[key], 'shape') else 'N/A'}")
    if len(model_state) > 5:
        print(f"  ... and {len(model_state) - 5} more keys")
    
    # Handle classifier size mismatch
    if 'classifier.weight' in model_state:
        if model_state['classifier.weight'].shape[0] != num_classes:
            print(f"\nWarning: Classifier size mismatch. Expected {num_classes} classes, "
                  f"but model has {model_state['classifier.weight'].shape[0]} classes.")
            print("Reinitializing classifier...")
            # Save the embedding size from the checkpoint
            if 'embedding.0.weight' in model_state:
                embedding_size = model_state['embedding.0.weight'].shape[0]
                print(f"Detected embedding size: {embedding_size}")
                # Reinitialize model with correct embedding size
                model = ResNetEmbedding(
                    backbone='resnet50',
                    emb_size=embedding_size,
                    num_classes=num_classes,
                    pretrained=False
                ).to(device)
            # Remove classifier weights to allow partial loading
            model_state.pop('classifier.weight', None)
            model_state.pop('classifier.bias', None)
    
    # Print model architecture
    print("\nModel architecture:")
    print(model)
    
    # Load state dict with strict=False to ignore missing keys
    try:
        model.load_state_dict(model_state, strict=False)
    except Exception as e:
        print(f"\nError loading model state: {str(e)}")
        print("\nAttempting to load only matching parameters...")
        # Try to load only the matching parameters
        model_dict = model.state_dict()
        # 1. Filter out unnecessary keys
        model_state = {k: v for k, v in model_state.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
        # 2. Update the model's state dict
        model_dict.update(model_state)
        # 3. Load the updated state dict
        model.load_state_dict(model_dict, strict=False)
    
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Print debug info
    print(f"Using device: {device}")
    print(f"Loading model from: {args.weights}")
    print(f"Training directory: {os.path.abspath(args.train_dir)}")
    
    # Check if training directory exists
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Training directory not found: {args.train_dir}")
    
    # Load model
    model = load_model(args.weights, device)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get class directories
    try:
        classes = [d for d in os.listdir(args.train_dir) 
                  if os.path.isdir(os.path.join(args.train_dir, d))]
        classes = sorted(classes)
        print(f"Found {len(classes)} class directories")
    except Exception as e:
        raise RuntimeError(f"Error reading training directory: {str(e)}")

    if not classes:
        # Try alternative directory structure (images directly in train_dir)
        print("No class subdirectories found. Checking for images directly in train directory...")
        image_files = [f for f in os.listdir(args.train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            print(f"Found {len(image_files)} images directly in train directory")
            # Create a single class
            classes = ['single_class']
        else:
            raise ValueError("No class directories or images found in training directory")

    # Compute centroids
    centroids = []
    valid_classes = []
    
    for cls in classes:
        if len(classes) > 1:  # If using class directories
            cls_dir = os.path.join(args.train_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Class directory not found: {cls_dir}")
                continue
            image_files = [f for f in os.listdir(cls_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        else:  # If using flat directory
            cls_dir = args.train_dir
            image_files = [f for f in os.listdir(cls_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nProcessing class: {cls} - Found {len(image_files)} images")
        
        if not image_files:
            print(f"Warning: No images found in {cls_dir}")
            continue
            
        emb_list = []
        processed_count = 0
        for fname in sorted(image_files)[:100]:  # Limit to 100 images per class
            try:
                img_path = os.path.join(cls_dir, fname)
                img = Image.open(img_path).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Updated embedding extraction
                    try:
                        # Try to get both logits and embeddings
                        outputs = model(x)
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            logits, emb = outputs
                        else:
                            # If model doesn't return embeddings, get them from the penultimate layer
                            features = model.backbone(x)
                            features = features.view(features.size(0), -1)  # Flatten
                            emb = model.embedding(features)
                        
                        # Ensure embedding is 1D
                        if len(emb.shape) > 2:
                            emb = emb.view(emb.size(0), -1)
                        emb = emb.squeeze(0).cpu().numpy()
                        emb = emb / (np.linalg.norm(emb) + 1e-12)
                        emb_list.append(emb)
                        processed_count += 1
                        
                        # Print progress
                        if processed_count % 10 == 0:
                            print(f"  Processed {processed_count}/{min(100, len(image_files))} images")
                            
                    except Exception as e:
                        print(f"\nError processing image {img_path}:")
                        print(f"  Error type: {type(e).__name__}")
                        print(f"  Error message: {str(e)}")
                        print(f"  Image shape: {np.array(img).shape}")
                        print(f"  Transformed shape: {x.shape}")
                        if 'emb' in locals():
                            print(f"  Embedding shape: {emb.shape if hasattr(emb, 'shape') else 'N/A'}")
                        continue
                            
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue
        
        if not emb_list:
            print(f"Warning: Could not process any images for class {cls}")
            continue
            
        # Compute class centroid
        centroid = np.mean(emb_list, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids.append(centroid)
        valid_classes.append(cls)
        print(f"  Computed centroid for class {cls} with {len(emb_list)} images")
    
    if not centroids:
        # Print directory structure for debugging
        print("\nDirectory structure:")
        for root, dirs, files in os.walk(args.train_dir):
            level = root.replace(args.train_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files[:5]:  # Show first 5 files
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        raise ValueError("No valid images found in any class directory")

    # Save results
    centroids = np.stack(centroids, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    result = {
        'centroids': centroids,
        'classes': valid_classes,
        'num_classes': len(valid_classes),
        'embedding_dim': centroids.shape[1]
    }
    
    torch.save(result, args.out)
    print(f"\nSuccessfully processed {len(valid_classes)} classes")
    print(f"Saved centroids to: {os.path.abspath(args.out)}")
    print(f"Embedding dimension: {centroids.shape[1]}")
    print(f"Available classes: {', '.join(valid_classes[:5])}{'...' if len(valid_classes) > 5 else ''}")

if __name__ == "__main__":
    main()