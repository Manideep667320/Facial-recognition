import cv2
import torch
import numpy as np
from model import ResNetEmbedding
from liveness_detector import LivenessDetector
from torchvision import transforms
from PIL import Image
import os
import json
import time
try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Using default configuration.")
    yaml = None

class VideoFaceRecognizer:
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize liveness detector (will work even if model is not available)
        liveness_model_path = config.get('liveness', {}).get('model_path', 'checkpoints/liveness_best.pth')
        self.liveness_detector = LivenessDetector(model_path=liveness_model_path, device=self.device)
        
        if not self.liveness_detector.is_available():
            print("Liveness detection is not available. All faces will be treated as real.")
        
        # Initialize face recognizer with the existing best.pth model
        model_path = os.path.join('checkpoints', 'best.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {os.path.abspath(model_path)}")
            
        # Handle class mapping and number of classes
        class_mapping_path = None
        class_mapping = {}
        
        # Try to get class mapping path from config or use default
        if 'class_mapping' in config:
            class_mapping_path = config['class_mapping']
        elif 'inference' in config and 'class_mapping' in config['inference']:
            class_mapping_path = config['inference']['class_mapping']
        else:
            # Try default paths
            default_paths = [
                'dataset/meta/class_index.json',
                'data/class_index.json',
                'class_index.json'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    class_mapping_path = path
                    break
        
        # Load class mapping if path exists
        if class_mapping_path and os.path.exists(class_mapping_path):
            try:
                with open(class_mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                print(f"Loaded {len(class_mapping)} classes from {class_mapping_path}")
            except Exception as e:
                print(f"Warning: Could not load class mapping from {class_mapping_path}: {e}")
        else:
            print(f"Warning: Class mapping file not found at {class_mapping_path or 'default locations'}")
        
        # Determine number of classes
        if 'num_classes' in config:
            num_classes = config['num_classes']
            print(f"Using {num_classes} classes from config")
        elif class_mapping:
            num_classes = len(class_mapping)
            print(f"Using {num_classes} classes from class mapping")
        else:
            num_classes = 31  # Default fallback
            print(f"Using default number of classes: {num_classes}")
            
        print(f"Loading face recognition model from {model_path}")
        self.face_recognizer = self._load_face_recognizer(
            model_path=model_path,
            num_classes=num_classes
        )
        
        # Initialize class names with the loaded mapping or empty dict
        self.class_names = {int(k): v for k, v in class_mapping.items()} if class_mapping else {}
        print(f"Initialized with {len(self.class_names)} class names")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _load_face_recognizer(self, model_path, num_classes):
        print(f"Initializing model with {num_classes} classes...")
        
        # Initialize the model with the correct parameters
        model = ResNetEmbedding(
            backbone='resnet50',  # or whatever backbone was used for training
            emb_size=512,         # embedding size
            num_classes=num_classes,
            pretrained=False      # Don't load ImageNet weights since we're loading our own
        )
        
        print(f"Loading weights from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Handle DataParallel model weights
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load the state dict with strict=False to handle mismatched keys
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            print("Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Trying to load with a different approach...")
            
            # Try to load just the weights that match
            try:
                model_dict = model.state_dict()
                # Filter out unnecessary keys
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
                model = model.to(self.device)
                model.eval()
                print("Model loaded successfully (partial weights)")
                return model
            except Exception as e2:
                print(f"Failed to load model: {str(e2)}")
                raise
        
    def _load_class_names(self, mapping_path):
        """Load class names from a JSON file.
        
        Args:
            mapping_path: Path to the JSON file containing class mappings.
                         Expected format: {"0": "class1", "1": "class2", ...}
        
        Returns:
            dict: A dictionary mapping class IDs to class names.
        """
        if not mapping_path or not os.path.exists(mapping_path):
            print(f"Warning: Class mapping file not found at {mapping_path}")
            return {}
            
        try:
            with open(mapping_path, 'r') as f:
                class_mapping = json.load(f)
                
            # Convert string keys to integers and handle different formats
            result = {}
            for k, v in class_mapping.items():
                try:
                    key = int(k)
                    result[key] = str(v)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert class ID {k} to integer: {e}")
                    continue
                    
            print(f"Loaded {len(result)} class names from {mapping_path}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse class mapping file {mapping_path}: {e}")
            return {}
        except Exception as e:
            print(f"Error loading class mapping from {mapping_path}: {e}")
            return {}
    
    def preprocess_face(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        return self.transform(face_img).unsqueeze(0).to(self.device)
    
    def recognize_face(self, face_img):
        with torch.no_grad():
            try:
                # Convert to tensor if it's a numpy array
                if isinstance(face_img, np.ndarray):
                    # Convert BGR to RGB if needed
                    if face_img.shape[2] == 3:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image and apply transforms
                    face_img = Image.fromarray(face_img)
                
                # Apply transforms and add batch dimension
                input_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = self.face_recognizer(input_tensor)
                
                # Handle different output formats
                if isinstance(outputs, (list, tuple)):
                    # If model returns (logits, embeddings) or [logits, ...]
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Ensure logits is 2D (batch_size, num_classes)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                # Get prediction and confidence
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get class name or default to 'Unknown'
                class_id = predicted.item()
                class_name = self.class_names.get(class_id, f"Unknown (ID: {class_id})")
                
                return class_name, confidence.item()
                
            except Exception as e:
                import traceback
                print(f"Error in face recognition: {str(e)}")
                print(traceback.format_exc())
                return "Error", 0.0
    
    def process_frame(self, frame):
        if frame is None or not isinstance(frame, np.ndarray):
            print("Error: Invalid frame received")
            return frame
            
        # Convert to grayscale for face detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert to uint8 if not already
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
                
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list if it's a numpy array
            if isinstance(faces, np.ndarray):
                faces = faces.tolist()
            else:
                faces = []
                
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            faces = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Check liveness if available
            is_live = True  # Default to True if liveness detection is not available
            if self.liveness_detector.is_available():
                is_live = self.liveness_detector.detect(frame, (x, y, w, h))
            
            if is_live:
                # Only recognize if the face is live
                name, confidence = self.recognize_face(face_roi)
                color = (0, 255, 0)  # Green for live
                label = f"{name} ({confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for spoof
                label = "Spoof detected!"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

def load_config():
    """Load configuration from config.yaml or use defaults."""
    default_config = {
        'model': {
            'backbone': 'resnet50',
            'embedding_size': 512,
            'pretrained': False
        },
        'inference': {
            'model_path': 'checkpoints/best.pth',
            'centroids_path': 'checkpoints/centroids.pth',
            'class_mapping': 'dataset/meta/class_index.json',
            'confidence_threshold': 0.8,
            'top_k': 5,
            'num_classes': 31
        },
        'liveness': {
            'enabled': True,
            'model_path': 'checkpoints/liveness_best.pth',
            'threshold': 0.85
        },
        'camera': {
            'device_id': 0,
            'width': 1280,
            'height': 720
        }
    }
    
    # Try to load from config.yaml if it exists
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                for key in user_config:
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}. Using default configuration.")
    
    return default_config

def check_opencv_gui():
    """Check if OpenCV GUI is working properly."""
    try:
        # Try to create a simple window
        test_window = 'test_window'
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        return True
    except Exception as e:
        print(f"OpenCV GUI check failed: {str(e)}")
        return False

def main():
    # Check if OpenCV GUI is working
    if not check_opencv_gui():
        print("""
        Error: OpenCV GUI is not working properly.
        This typically happens when OpenCV is not built with GUI support.
        
        Try one of these solutions:
        1. Install OpenCV with GUI support:
           pip uninstall opencv-python
           pip install opencv-python-headless  # For headless environments
           OR
           pip install opencv-python           # For systems with GUI support
           
        2. If you're using a remote server, make sure to enable X11 forwarding.
        """)
        return
    
    # Load configuration
    config = load_config()
    
    print("\n" + "="*50)
    print("Face Recognition System")
    print("="*50)
    print("Configuration:")
    print(f"- Model: {config.get('inference', {}).get('model_path', 'checkpoints/best.pth')}")
    print(f"- Classes: {config.get('inference', {}).get('class_mapping', 'dataset/meta/class_index.json')}")
    print(f"- Liveness Detection: {'Enabled' if config.get('liveness', {}).get('enabled', False) else 'Disabled'}")
    print("="*50 + "\n")
    
    try:
        # Initialize recognizer
        recognizer = VideoFaceRecognizer(config)
        
        # Initialize video capture
        cap = cv2.VideoCapture(config['camera']['device_id'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
        
        if not cap.isOpened():
            raise RuntimeError("Could not open video device")
        
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 'f' to toggle fullscreen")
        print("\nStarting video capture...")
        
        window_name = 'Face Recognition (Press q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        fullscreen = False
        
        # For FPS calculation
        prev_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time + 1e-6))
            prev_time = current_time
            
            # Process frame
            processed_frame = recognizer.process_frame(frame)
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {int(fps)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display result
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
            cv2.imshow(window_name, processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = not fullscreen
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("\nVideo capture stopped")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
