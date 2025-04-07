# Coral Segmentation with SAM2

This project implements coral segmentation using the SAM2 (Segment Anything Model 2) architecture. It includes data processing, training, and visualization tools for coral annotation datasets.

## Project Structure

```
.
├── config/                    # Training configuration files
│   └── train_sam2.yaml       # Main training configuration
├── data/                     # Raw data directory
│   ├── images/              # Raw images
│   └── annotations/         # Raw annotations
├── processed_data/          # Processed dataset
│   ├── train/              # Training data
│   │   ├── images/        # Training images
│   │   └── annotations/   # Training annotations
│   └── validate/          # Validation data
│       ├── images/        # Validation images
│       └── annotations/   # Validation annotations
├── sam2/                   # SAM2 model implementation
├── scripts/                # Utility scripts
│   ├── prepare_data.py    # Data processing script
│   └── visualize_annotations.py  # Visualization script
├── outputs/               # Training outputs
├── checkpoints/          # Model checkpoints
└── logs/                # Training logs
```

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   ```bash
   # Process the dataset
   python prepare_data.py
   ```

3. **Visualization**
   ```bash
   # Visualize annotations
   python visualize_annotations.py
   ```

## Training Configuration

The training configuration is specified in `config/train_sam2.yaml`. Key settings include:

- Model: sam2.1_hiera_small
- Learning rate: 0.001
- Batch size: 16
- Epochs: 10
- Optimizer: Adam

## GPU Requirements

- NVIDIA GPU with CUDA support (recommended: A100)
- Minimum 16GB VRAM for training
- CUDA 11.7 or later

## Scaling to Full Dataset

For the full dataset (30,000 images), consider:

1. **Storage Options**:
   - Cloud storage (AWS S3, Google Cloud Storage)
   - Network-attached storage (NAS)
   - High-performance local storage

2. **Training Considerations**:
   - Adjust batch size based on GPU memory
   - Use data parallelism for multi-GPU training
   - Implement checkpointing for long training runs

## License

[Add your license information here]

## Contact

[Add contact information here] 