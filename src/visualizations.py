import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import FolderDataset, get_valid_transform
from model import ResNetEmbedding

def plot_training_metrics(log_dir='runs/experiment_1'):
    """Plot training and validation metrics from TensorBoard logs"""
    writer = SummaryWriter(log_dir=log_dir)
    
    # Add dummy data for demonstration
    # In practice, these would be read from your training logs
    for epoch in range(1, 11):
        writer.add_scalar('Loss/train', 1.5 - epoch*0.1, epoch)
        writer.add_scalar('Loss/val', 1.4 - epoch*0.09, epoch)
        writer.add_scalar('Accuracy/train', 0.1 + epoch*0.08, epoch)
        writer.add_scalar('Accuracy/val', 0.1 + epoch*0.07, epoch)
    
    # This would normally be done during training
    # For now, we'll just show how to access the data
    print(f"TensorBoard logs available at: {os.path.abspath(log_dir)}")
    print("Run 'tensorboard --logdir=runs' to view detailed metrics")
    writer.close()

def visualize_embeddings(model, dataloader, class_names, output_dir='visualizations'):
    """Visualize embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    
    device = next(model.parameters()).device
    model.eval()
    
    # Extract embeddings and labels
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            _, emb = model(images, return_embedding=True)
            embeddings.append(emb.cpu())
            labels_list.append(labels)
    
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels_list).numpy()
    
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.title('t-SNE Visualization of Face Embeddings')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tsne_embeddings.png'))
    print(f"t-SNE plot saved to {os.path.join(output_dir, 'tsne_embeddings.png')}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir='visualizations'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def visualize_predictions(model, dataloader, class_names, output_dir='visualizations', num_images=9):
    """Visualize model predictions on sample images"""
    device = next(model.parameters()).device
    model.eval()
    
    # Get a batch of data
    images, labels, _ = next(iter(dataloader))
    images = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Convert to numpy for visualization
    images = images.cpu()
    
    # Plot
    plt.figure(figsize=(15, 15))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i+1)
        
        # Convert from CxHxW to HxWxC for matplotlib
        img = images[i].permute(1, 2, 0).numpy()
        # Undo normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = std * img + mean
        img = img.clip(0, 1)
        
        # Show image
        plt.imshow(img)
        
        # Set title with true and predicted labels
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        color = 'green' if labels[i] == preds[i] else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    print(f"Sample predictions saved to {os.path.join(output_dir, 'sample_predictions.png')}")

def main():
    # Configuration
    ROOT_DIR = Path(__file__).parent.parent
    data_dir = ROOT_DIR / 'data'
    img_size = 224
    batch_size = 32
    
    # Load class names
    with open(ROOT_DIR / 'dataset' / 'meta' / 'class_index.json', 'r') as f:
        class_index = json.load(f)
    idx_to_class = {int(k): v for k, v in class_index.items()}
    
    # Create validation dataset and dataloader
    val_dataset = FolderDataset(
        root=os.path.join(data_dir, 'val'),
        transform=get_valid_transform(img_size=img_size)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model and load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with correct number of classes
    model = ResNetEmbedding(
        backbone='resnet50',
        emb_size=512,
        num_classes=len(idx_to_class),
        pretrained=False
    ).to(device)
    
    # Load the trained model weights
    weights_path = ROOT_DIR / 'checkpoints' / 'best.pth'
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device)
        model_state = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded model weights from {weights_path}")
    else:
        print(f"Warning: Model weights not found at {weights_path}")
    
    model.eval()
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Plot training metrics (requires training logs)
    # plot_training_metrics()
    
    # 2. Visualize embeddings
    # visualize_embeddings(model, val_loader, 
    #                     [idx_to_class[i] for i in range(len(idx_to_class))])
    
    # 3. Visualize sample predictions
    visualize_predictions(model, val_loader, 
                         [idx_to_class[i] for i in range(len(idx_to_class))])
    
    print("Visualization generation complete!")

if __name__ == "__main__":
    main()
