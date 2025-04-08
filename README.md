# Coral Reef Segmentation with SAM2.1

This project fine-tunes the Segment Anything Model (SAM2.1) specifically for coral reef segmentation, helping it distinguish between coral and non-coral elements (like rocks).

## Project Structure

```
.
├── training/                    # Training pipeline
│   ├── config/                 # Training configuration files
│   ├── data/                   # Training data (images + annotations)
│   │   ├── train/             # Training images and annotations
│   │   └── validate/          # Validation images and annotations
│   ├── scripts/               # Training-related scripts
│   │   ├── prepare_data.py    # Data preparation for training
│   │   └── train.py          # Training script
│   └── checkpoints/           # Saved model checkpoints
│
└── inference/                  # Inference pipeline
    ├── config/                # Inference configuration
    ├── data/                  # Unlabeled images for prediction
    ├── scripts/              # Inference-related scripts
    │   └── predict.py        # Prediction script
    └── outputs/              # Generated masks and predictions
```

## Workflows

### 1. Training Pipeline

The training pipeline fine-tunes SAM2.1 on labeled coral data to improve its ability to distinguish coral from non-coral elements.

#### Setup
1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare training data:
   ```bash
   cd training
   python scripts/prepare_data.py
   ```

3. Start training:
   ```bash
   python scripts/train.py --config config/train_sam2.yaml
   ```

#### Data Requirements
- Training images (30K+ photos)
- Corresponding JSON annotations
- Validation set (recommended 10% of data) for:
  - Monitoring training progress
  - Preventing overfitting
  - Verifying the model's ability to distinguish coral from non-coral elements
  - Early stopping and hyperparameter tuning

Note: While SAM2.1 is already a robust model, the validation set is particularly important for fine-tuning as it helps ensure the model is learning to focus on coral-specific features rather than overfitting to the training data.

### 2. Inference Pipeline

The inference pipeline uses the fine-tuned SAM2.1 model to generate masks for unlabeled coral images.

#### Setup
1. Ensure you have a trained model checkpoint in `training/checkpoints/`
2. Place unlabeled images in `inference/data/`
3. Configure inference settings in `inference/config/inference.yaml`

#### Usage
```bash
cd inference
# For full inference on all images
python scripts/predict.py --config config/inference.yaml --checkpoint ../training/checkpoints/model.pt

# To test the pipeline with a single image
python scripts/test_inference.py
```

#### Process Details
1. The inference script will:
   - Load your fine-tuned SAM2.1 model
   - Process each image in the input directory
   - Generate masks using SAM2.1's automatic mask generation
   - Select the best mask based on confidence scores
   - Save masks as PNG files in the output directory

2. Supported image formats:
   - JPG/JPEG
   - PNG
   - TIFF

3. Output:
   - Each input image will generate a corresponding mask file
   - Mask files are saved as `{original_filename}_mask.png`
   - Masks are binary (black and white) images where white represents coral regions

#### Testing
Before running inference on your full dataset:
1. Place a test image in `inference/test_data/input/test_image.jpg`
2. Run `python scripts/test_inference.py`
3. Check the generated mask in `inference/test_data/output/test_mask.png`

This will help verify that:
- The model loads correctly
- Image processing works as expected
- Mask generation produces reasonable results

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full list

## Notes
- Training requires significant GPU memory (recommended: NVIDIA A100)
- Inference can be run on CPU but will be slower
- For large datasets, consider using cloud storage for data

## License

[Add your license information here]

## Contact

[Add contact information here] 