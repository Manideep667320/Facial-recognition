# src/infer.py
"""
Inference script using centroids saved by compute_centroids.py

Usage:
 python src/infer.py --test_dir data/test --weights checkpoints/best.pth \
    --centroids checkpoints/centroids.pth --mapping checkpoints/class_mapping.json \
    --out submission.json --img_size 224
"""
import os
import argparse
import json
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from model import ResNetEmbedding

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--centroids', required=True)
    p.add_argument('--mapping', required=True, help='JSON file mapping class index to class name (or class name list)')
    p.add_argument('--out', default='submission.json')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def load_model(weights_path, device, num_classes=31):
    """
    Load model from checkpoint with proper handling of class size mismatches
    Args:
        weights_path: Path to the model checkpoint
        device: Device to load the model on
        num_classes: Number of classes in the current task (default: 31)
    """
    # First, get the embedding size from the checkpoint if available
    ckpt = torch.load(weights_path, map_location='cpu')
    model_state = ckpt.get('model_state', ckpt)
    
    # Try to determine the correct embedding size from the checkpoint
    emb_size = 512  # Default
    if 'embedding.0.weight' in model_state:
        emb_size = model_state['embedding.0.weight'].shape[0]
        print(f"Detected embedding size from checkpoint: {emb_size}")
    
    # Initialize model with the detected embedding size
    model = ResNetEmbedding(
        backbone='resnet50',
        emb_size=emb_size,
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    
    # Handle classifier size mismatch
    if 'classifier.weight' in model_state:
        if model_state['classifier.weight'].shape[0] != num_classes:
            print(f"Warning: Classifier size mismatch. Expected {num_classes} classes, "
                  f"but model has {model_state['classifier.weight'].shape[0]} classes.")
            print("Loading only backbone and embedding weights, reinitializing classifier...")
            # Remove classifier weights to allow partial loading
            model_state.pop('classifier.weight', None)
            model_state.pop('classifier.bias', None)
    
    # Print model state keys for debugging
    print("\nModel state keys:")
    for key in list(model_state.keys())[:5]:
        print(f"  {key}: {model_state[key].shape if hasattr(model_state[key], 'shape') else 'N/A'}")
    if len(model_state) > 5:
        print(f"  ... and {len(model_state) - 5} more keys")
    
    # Load the state dict with strict=False to ignore missing keys
    model.load_state_dict(model_state, strict=False)
    
    # If classifier wasn't loaded, initialize it with the correct input size
    if 'classifier.weight' not in model_state:
        print("Initializing new classifier layer...")
        # Get the correct input size from the embedding layer
        classifier_input_size = model.embedding[0].out_features
        print(f"Initializing classifier with input size: {classifier_input_size}, num_classes: {num_classes}")
        model.classifier = nn.Linear(
            classifier_input_size, 
            num_classes
        ).to(device)
    
    model.eval()
    return model

def load_mapping(mapping_path):
    with open(mapping_path, 'r') as f:
        data = json.load(f)
    
    # If it's a dictionary with numeric keys, convert to list
    if isinstance(data, dict):
        # Check if keys are numeric strings
        if all(k.isdigit() for k in data.keys() if k != '0'):
            max_idx = max(int(k) for k in data.keys())
            arr = [""] * (max_idx + 1)
            for k, v in data.items():
                if k.isdigit():
                    arr[int(k)] = v
            return arr
        return data  # Return as is if not numeric keys
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("mapping JSON must be a dict or list")

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load class mapping
    with open(args.mapping, 'r') as f:
        mapping = json.load(f)
    
    # Determine number of classes from mapping
    if isinstance(mapping, dict):
        num_classes = len(mapping)
        # Convert dict to list if keys are numeric indices
        if all(k.isdigit() for k in mapping.keys()):
            max_idx = max(int(k) for k in mapping.keys()) + 1
            class_list = [''] * max_idx
            for k, v in mapping.items():
                class_list[int(k)] = v
            mapping = class_list
    elif isinstance(mapping, list):
        num_classes = len(mapping)
    else:
        raise ValueError(f"Unexpected mapping format: {type(mapping)}")
    
    print(f"Loaded mapping with {num_classes} classes")
    
    # Load model with correct number of classes
    model = load_model(args.weights, device, num_classes=num_classes)
    
    # Load centroids
    print("Loading centroids...")
    centroids_data = torch.load(args.centroids, map_location='cpu')
    
    # Handle different centroid formats
    if isinstance(centroids_data, dict):
        centroids_np = centroids_data['centroids']
        if 'classes' in centroids_data:
            centroid_classes = centroids_data['classes']
        else:
            centroid_classes = list(range(len(centroids_np)))
    else:
        centroids_np = centroids_data
        centroid_classes = list(range(len(centroids_np)))
    
    # Convert to numpy array if it's a tensor
    if isinstance(centroids_np, torch.Tensor):
        centroids_np = centroids_np.numpy()
    
    # Ensure centroids are float32 and normalized
    centroids_np = centroids_np.astype(np.float32)
    norms = np.linalg.norm(centroids_np, axis=1, keepdims=True) + 1e-12
    centroids_norm = centroids_np / norms
    
    # Convert to tensor for GPU if available
    centroids_tensor = torch.from_numpy(centroids_norm).to(device)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = []
    image_files = [f for f in os.listdir(args.test_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images in {args.test_dir}")
    
    for fname in tqdm(sorted(image_files), desc="Processing images"):
        try:
            path = os.path.join(args.test_dir, fname)
            img = Image.open(path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, emb = model(x, return_embedding=True)
            
            # Get embedding and normalize
            e = emb.cpu().numpy()[0]
            e = e / (np.linalg.norm(e) + 1e-12)
            
            # Calculate similarities with centroids
            sims = centroids_norm @ e  # (C,)
            idx = int(np.argmax(sims))
            
            # Get class name from mapping
            if isinstance(mapping, (list, tuple)) and idx < len(mapping):
                class_name = mapping[idx]
            elif isinstance(mapping, dict):
                class_name = mapping.get(str(idx), str(idx))
            else:
                class_name = str(idx)
                
            results.append({
                'image_name': fname,
                'class': class_name,
                'class_idx': int(idx),
                'confidence': float(sims[idx])
            })
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue
    
    # Save results
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {os.path.abspath(args.out)}")
    print(f"Successfully processed {len(results)}/{len(image_files)} images")

if __name__ == "__main__":
    main()
