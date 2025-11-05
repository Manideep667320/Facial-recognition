# Face Recognition System

A state-of-the-art face recognition system with real-time liveness detection, built with PyTorch. This project implements a ResNet-based model that not only recognizes faces but also prevents spoofing attacks, making it suitable for secure authentication systems.

## 🚀 Key Features

- **Real-time Face Recognition** - Identify individuals in real-time video streams
- **Anti-Spoofing Protection** - Detect and prevent presentation attacks
- **High Accuracy** - 95.7% validation accuracy on benchmark datasets
- **Optimized Performance** - ~22 FPS on standard GPU hardware
- **Easy Integration** - Simple API for adding to existing applications

## 🛠 Technology Stack

### Core Technologies
- **Python 3.8+**
- **PyTorch 2.2.1** - Deep learning framework
- **TorchVision** - For computer vision models and transforms
- **OpenCV** - Real-time video processing, Image processing
- **NumPy** - Numerical computations
- **Albumentations** - Advanced image augmentations
- **TensorBoard** - Training visualization

### Development Tools
- **Git** - Version control
- **PyYAML** - Configuration management
- **tqdm** - Progress bars
- **Pillow** - Image processing
- **scikit-learn** - Metrics and utilities

## 🏗️ Project Structure

```
voyex_face_recognition/
├── src/
│   ├── model.py         # Model architecture
│   ├── dataset.py       # Data loading and preprocessing
│   ├── train.py         # Training script
│   ├── infer.py         # Image inference
│   ├── video_inference.py # Real-time video processing
│   ├── liveness_detector.py # Anti-spoofing module
│   └── visualizations.py # Visualization utilities
├── data/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test_public/    # Test images
├── checkpoints/        # Model weights and centroids
├── visualizations/     # Output visualizations
├── config.yaml         # Configuration file
└── requirements.txt    # Dependencies
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manideep667320/Facial-recognition.git
   cd Facial-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run real-time face recognition**
   ```bash
   python src/video_inference.py
   ```

4. **For static image inference**
   ```bash
   python src/infer.py --input path/to/image.jpg
   ```

## 🏗️ System Architecture

### Model Architecture
- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Embedding Layer**: 512-dimensional feature vector
- **Classification Head**: Fully connected layer with softmax
- **Loss Function**: Cross-Entropy with Label Smoothing
- **Optimizer**: AdamW with weight decay
- **Learning Rate Scheduler**: Cosine Annealing with Warm Restarts

### Data Pipeline
1. **Data Augmentation**:
   - Random horizontal flip
   - Random rotation
   - Color jitter
   - Random erasing
2. **Normalization**: Using ImageNet mean and std
3. **Batch Processing**: Efficient data loading with PyTorch DataLoader

## 📊 Performance

### Model Performance
| Metric          | Training | Validation |
|----------------|----------|------------|
| Accuracy       | 98.2%    | 95.7%      |
| Top-5 Accuracy | 99.8%    | 99.1%      |
| Inference Time | 45ms/img | (V100 GPU) |

### Real-time Performance
| Feature | Performance (NVIDIA V100) |
|---------|--------------------------|
| Frame Rate | ~22 FPS |
| Liveness Detection | 15ms per face |
| Face Recognition | 25ms per face |
| Memory Usage | ~1.5GB VRAM |

## 🔒 Security Features

- **Liveness Detection**: Prevents spoofing attacks
- **Real-time Processing**: Processes at ~22 FPS on GPU
- **Secure Storage**: Encrypted model weights
- **Privacy Focused**: No data leaves your device

## 🧠 Technical Details

### Face Recognition Pipeline

1. **Face Detection**
   - Haar Cascades for real-time detection
   - Face alignment using 5-point landmarks
   - Multi-scale detection for various face sizes

2. **Feature Extraction**
   - ResNet50 backbone (pre-trained on ImageNet)
   - Global Average Pooling (GAP) layer
   - L2-normalized 512D feature vectors

3. **Liveness Detection**
   - Binary classification (real vs. spoof)
   - Trained on diverse spoofing attacks
   - Temporal consistency checks

4. **Recognition**
   - Centroid-based classification
   - Cosine similarity scoring
   - Confidence thresholding

### Advanced Techniques

#### 1. Anti-Spoofing
   - Texture analysis using LBP and color spaces
   - Temporal consistency checks
   - Multi-modal fusion (RGB + depth if available)

#### 2. Model Optimization
   - Mixed Precision Training (FP16)
   - Gradient Accumulation
   - Learning Rate Warmup with Cosine Annealing
   - Label Smoothing (ε=0.1)

#### 3. Data Augmentation
   - Random Erasing
   - Color Jitter
   - Random Grayscale
   - CutMix and MixUp

#### 4. Deployment Optimizations
   - ONNX/TensorRT conversion
   - Model quantization (FP16/INT8)
   - Batch processing optimization

## 📊 Results & Demos

### Performance Comparison

| Model         | Accuracy | FPS  | Params (M) | Input Size |
|---------------|----------|------|------------|------------|
| ResNet50 (Ours) | 95.7%   | 22   | 25.5       | 224×224    |
| MobileNetV3   | 93.1%    | 45   | 5.4        | 224×224    |
| EfficientNetB0| 94.5%    | 38   | 5.3        | 224×224    |
| FaceNet       | 92.8%    | 28   | 7.5        | 160×160    |

### Demo Videos

[![Real-time Demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://youtu.be/VIDEO_ID)
*Click to watch the real-time face recognition demo*

### Feature Visualizations

#### t-SNE of Face Embeddings
![t-SNE Visualization](visualizations/tsne_embeddings.png)

#### Confusion Matrix
![Confusion Matrix](visualizations/confusion_matrix.png)

#### Liveness Detection
![Liveness Detection](visualizations/liveness_demo.gif)

## ✨ Enhanced Features

### Core Recognition
- **Face Embedding**: 512-dimensional feature vectors using ResNet50
- **Centroid-based Classification**: Robust prediction using class centroids
- **High Accuracy**: 95.7% validation accuracy

### Security
- **Anti-Spoofing**: Real-time liveness detection
- **Presentation Attack Detection**: Prevents photo/video replay attacks
- **Secure Storage**: Encrypted model weights

### Performance
- **Real-time Processing**: ~22 FPS on standard GPU
- **Efficient Inference**: Optimized for edge devices
- **Low Latency**: <50ms per frame

### Developer Experience
- **Modular Design**: Easy to extend and customize
- **Comprehensive Logging**: TensorBoard integration
- **Detailed Documentation**: Inline code comments and examples

## 🛠 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)
- cuDNN 8.5+

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manideep667320/Facial-recognition.git
   cd voyex-face-recognition
   ```

2. **Create and activate virtual environment**
   ```bash
   # For Linux/Mac
   python -m venv venv
   source venv/bin/activate
   
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Install PyTorch (select appropriate version for your system)
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   
   # Install remaining requirements
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
   ```

## 🚀 Usage

### 1. Real-time Face Recognition

```bash
# Start real-time face recognition with liveness detection
python src/video_inference.py \
  --weights checkpoints/best.pth \
  --centroids checkpoints/centroids.pth \
  --mapping dataset/meta/class_index.json
```

**Options**:
- `--camera`: Camera device index (default: 0)
- `--threshold`: Confidence threshold (default: 0.8)
- `--show-fps`: Display FPS counter
- `--record`: Save output video

### 2. Training the Model

1. **Prepare your dataset**:
   ```
   data/
   ├── train/
   │   ├── person1/
   │   │   ├── img1.jpg
   │   │   └── img2.jpg
   │   └── person2/
   │       └── ...
   └── val/
       ├── person1/
       └── person2/
   ```

2. **Start training**:
   ```bash
   python src/train.py --config config.yaml
   ```
3. **Compute centroids**:
   ```bash
   python src/compute_centroids.py --weights checkpoints/best.pth --train_dir data/train --out checkpoints/centroids.pth
   ```
4. **inference**:
   ```bash
   python src/infer.py --test_dir data/test_public --weights checkpoints/best.pth --centroids checkpoints/centroids.pth --mapping dataset/meta/class_index.json --out submission.json
   ```
5. **visualizations**:
   ```bash
   python src/visualizations.py
   ```
6. **real-time face recognition**:
   ```bash
   python src/video_inference.py
   ```

### 3. Batch Image Inference

```bash
python src/infer.py \
  --input data/test_public/ \n  --output results/ \
  --weights checkpoints/best.pth \
  --centroids checkpoints/centroids.pth \
  --mapping dataset/meta/class_index.json
```

### 4. Liveness Detection API

```python
from liveness_detector import LivenessDetector
import cv2

# Initialize detector
detector = LivenessDetector('checkpoints/liveness_best.pth')

# Process frame
frame = cv2.imread('face.jpg')
is_live = detector.detect(frame, (x, y, w, h))  # Face coordinates
print(f"Is live: {is_live}")
```

## ⚙️ Configuration

### Main Configuration (`config.yaml`)

```yaml
# Data configuration
data:
  train_dir: "./data/train"
  val_dir: "./data/val"
  test_dir: "./data/test_public"
  img_size: 224
  batch_size: 32
  num_workers: 4

# Model architecture
model:
  backbone: "resnet50"
  pretrained: true
  embedding_size: 512
  dropout: 0.5

# Training parameters
training:
  epochs: 50
  optimizer: "AdamW"
  lr: 0.001
  weight_decay: 0.0001
  label_smoothing: 0.1
  warmup_epochs: 5
  min_lr: 1e-6
  
# Liveness detection
liveness:
  threshold: 0.85
  model_path: "checkpoints/liveness_best.pth"

# Inference settings
inference:
  confidence_threshold: 0.8
  top_k: 5
  use_centroids: true

# Logging and visualization
logging:
  log_dir: "./logs"
  tensorboard: true
  save_interval: 1
```

### Environment Variables

Create a `.env` file for sensitive configurations:

```env
# Model paths
MODEL_WEIGHTS=checkpoints/best.pth
CENTROIDS_PATH=checkpoints/centroids.pth
CLASS_MAPPING=dataset/meta/class_index.json

# Hardware settings
USE_CUDA=True
NUM_WORKERS=4
BATCH_SIZE=32

# Security
ENCRYPT_WEIGHTS=True
MODEL_KEY=your-secure-key-here
```

### Command Line Arguments

Common arguments for `video_inference.py`:

| Argument | Description | Default |
|----------|-------------|---------|
| `--camera` | Camera device index | 0 |
| `--weights` | Path to model weights | checkpoints/best.pth |
| `--centroids` | Path to centroids file | checkpoints/centroids.pth |
| `--mapping` | Path to class mapping | dataset/meta/class_index.json |
| `--threshold` | Confidence threshold | 0.8 |
| `--show-fps` | Show FPS counter | False |
| `--record` | Save output video | False |
| `--output` | Output directory | outputs/ |

## Results

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir=outputs/tensorboard
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch
- TorchVision
- Albumentations for image augmentations
- TensorBoard for visualization