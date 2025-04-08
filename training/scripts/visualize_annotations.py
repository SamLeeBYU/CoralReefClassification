import os
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

def decode_rle(rle_data, target_height, target_width):
    """Decode RLE format to binary mask and resize if necessary."""
    if isinstance(rle_data, dict) and 'counts' in rle_data and 'size' in rle_data:
        try:
            rle = {'counts': rle_data['counts'].encode('utf-8'), 'size': rle_data['size']}
            mask = mask_utils.decode(rle)
            if mask.shape[:2] != (target_height, target_width):
                mask = cv2.resize(mask.astype(np.uint8), (target_width, target_height), 
                                interpolation=cv2.INTER_NEAREST)
            return mask.astype(bool)
        except Exception as e:
            print(f"Error decoding RLE: {e}")
            return None
    return None

def visualize_annotations(data_dir: str, num_samples: int = 5):
    """Visualize annotations for a few samples."""
    # Load the dataset
    dataset_path = os.path.join(data_dir, 'annotations', 'dataset.json')
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Get random samples
    np.random.seed(42)  # for reproducibility
    sample_indices = np.random.choice(len(dataset['images']), min(num_samples, len(dataset['images'])), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(len(sample_indices), 2, figsize=(15, 5*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_idx in enumerate(sample_indices):
        img_info = dataset['images'][img_idx]
        img_path = os.path.join(data_dir, 'images', img_info['file_name'])
        
        # Load and display original image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Create mask overlay
        mask_overlay = np.zeros_like(img)
        
        # Get annotations for this image
        img_anns = [ann for ann in dataset['annotations'] if ann['image_id'] == img_info['id']]
        
        # Use different colors for different instances
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        
        for ann_idx, ann in enumerate(img_anns):
            mask = decode_rle(ann['segmentation'], height, width)
            if mask is not None:
                color = colors[ann_idx % len(colors)]
                mask_overlay[mask] = color
        
        # Display original image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Original Image ({width}x{height})')
        axes[idx, 0].axis('off')
        
        # Overlay masks on image
        alpha = 0.3
        overlay = cv2.addWeighted(img, 1, mask_overlay, alpha, 0)
        axes[idx, 1].imshow(overlay)
        axes[idx, 1].set_title(f'With Annotations ({len(img_anns)} instances)')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('annotation_visualization.png')
    plt.close()
    print(f"Processed {len(sample_indices)} images with annotations")

def main():
    data_dir = 'processed_data'
    print("Visualizing annotations...")
    visualize_annotations(data_dir)
    print("Visualization saved as 'annotation_visualization.png'")

if __name__ == "__main__":
    main() 