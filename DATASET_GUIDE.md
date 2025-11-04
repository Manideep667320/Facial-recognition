# Voyex Face Recognition - Dataset Guide

This guide explains how to prepare and use the dataset for the Voyex Face Recognition project.

## 1. Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── images/                    # Raw images
│   ├── person1/              # Each person has their own folder
│   │   ├── img1.jpg          # Multiple images per person
│   │   └── img2.jpg
│   └── person2/
│       └── ...
└── meta/
    ├── class_index.json      # Maps class IDs to person names
    └── ...                   # Other metadata files
```

## 2. Setting Up the Dataset

1. **Prepare your images**:
   - Place face images in the `dataset/images/` directory
   - Each person should have their own subfolder
   - Supported formats: JPG, JPEG, PNG

2. **Run the preparation script**:
   ```bash
   python prepare_dataset.py
   ```
   This will:
   - Create the necessary directory structure in `data/`
   - Split images into training and validation sets
   - Print a summary of the dataset

3. **Verify the dataset**:
   ```bash
   python verify_dataset.py
   ```
   This will:
   - Load the dataset
   - Show sample images with labels
   - Save a sample image as `dataset_sample.png`

## 3. Dataset Usage in Training

The training script (`train.py`) will automatically use the dataset from the `data/` directory:

```bash
python src/train.py --config config.yaml
```

## 4. Adding New People

1. Create a new folder in `dataset/images/` with the person's name
2. Add their images to the folder
3. Update the `class_index.json` file with the new person's ID and name
4. Run the preparation script again

## 5. Dataset Statistics

After preparing the dataset, you can check the distribution of images:

```
python -c "
import json
from pathlib import Path

# Load class index
with open('dataset/meta/class_index.json', 'r') as f:
    class_index = json.load(f)

# Count images in each split
for split in ['train', 'val']:
    print(f'\n{split.upper()}:')
    split_dir = Path('data') / split
    for class_id, class_name in class_index.items():
        class_dir = split_dir / class_name
        if class_dir.exists():
            num_images = len(list(class_dir.glob('*.*')))
            print(f'  {class_name}: {num_images} images')
"
```

## 6. Troubleshooting

- **No images found**: Make sure images are in the correct format (.jpg, .jpeg, .png)
- **Class not found**: Check for typos in folder names and `class_index.json`
- **Memory issues**: Reduce batch size in `config.yaml` if you run out of memory
