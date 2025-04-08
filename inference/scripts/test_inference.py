import os
import sys
import torch
from pathlib import Path
import yaml
from predict import load_model, process_image, save_mask

def test_inference():
    """Test the inference pipeline with a sample image."""
    print("Testing inference pipeline...")
    
    # Create test directories
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "input").mkdir(exist_ok=True)
    (test_dir / "output").mkdir(exist_ok=True)
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load model
        print("Loading model...")
        checkpoint_path = "../training/checkpoints/model.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
            
        predictor = load_model(checkpoint_path, device)
        print("Model loaded successfully")
        
        # Process test image
        test_image_path = test_dir / "input" / "test_image.jpg"
        if not test_image_path.exists():
            print("Please place a test image at test_data/input/test_image.jpg")
            return
            
        print("Processing test image...")
        mask = process_image(str(test_image_path), predictor, device)
        
        # Save result
        output_path = test_dir / "output" / "test_mask.png"
        save_mask(mask, str(output_path))
        
        print(f"Test completed successfully!")
        print(f"Input image: {test_image_path}")
        print(f"Output mask: {output_path}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(test_inference()) 