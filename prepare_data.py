import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from typing import Dict, List, Tuple

def validate_annotation(ann: Dict) -> bool:
    """Validate a single annotation entry."""
    required_fields = ['id', 'image_id', 'area', 'segmentation', 'bbox']
    return all(field in ann for field in required_fields)

def validate_image_info(img: Dict) -> bool:
    """Validate image information."""
    required_fields = ['file_name', 'height', 'width']
    return all(field in img for field in required_fields)

def process_dataset(data_dir: str, output_dir: str, dataset_type: str) -> Tuple[List[Dict], List[Dict]]:
    """Process the dataset and prepare it for SAM2 training."""
    images_dir = os.path.join(data_dir, 'images')
    jsons_dir = os.path.join(data_dir, 'jsons')
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, dataset_type, 'images')
    output_annotations_dir = os.path.join(output_dir, dataset_type, 'annotations')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)
    
    # Initialize COCO-style dataset structure
    dataset = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'coral'}]
    }
    
    # Process each JSON file and maintain image-annotation correspondence
    json_files = [f for f in os.listdir(jsons_dir) if f.endswith('.json') and f != 'consolidated_annotations.json']
    
    # First, create a mapping from image filename to new image ID
    filename_to_id = {}
    current_img_id = 0
    current_ann_id = 0
    
    print(f"\nProcessing {dataset_type} dataset...")
    for json_file in tqdm(json_files, desc=f"Processing {dataset_type} files"):
        json_path = os.path.join(jsons_dir, json_file)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Validate image info
        if not validate_image_info(data['image']):
            print(f"Warning: Invalid image info in {json_file}")
            continue
        
        img_name = data['image']['file_name']
        
        # Skip if we've already processed this image
        if img_name in filename_to_id:
            continue
            
        # Copy image to output directory
        src_img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(src_img_path):
            print(f"Warning: Image file not found: {src_img_path}")
            continue
            
        dst_img_path = os.path.join(output_images_dir, img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # Assign new image ID and store mapping
        filename_to_id[img_name] = current_img_id
        
        # Update image info with new ID
        image_info = data['image'].copy()
        image_info['id'] = current_img_id
        dataset['images'].append(image_info)
        
        # Process annotations for this image
        for ann in data['annotations']:
            if not validate_annotation(ann):
                print(f"Warning: Invalid annotation in {json_file}")
                continue
                
            # Create new annotation with updated IDs
            new_ann = ann.copy()
            new_ann['id'] = current_ann_id
            new_ann['image_id'] = current_img_id
            new_ann['category_id'] = 1
            dataset['annotations'].append(new_ann)
            current_ann_id += 1
        
        current_img_id += 1
    
    # Save the processed dataset
    output_json = os.path.join(output_annotations_dir, 'dataset.json')
    with open(output_json, 'w') as f:
        json.dump(dataset, f)
    
    print(f"\n{dataset_type} dataset statistics:")
    print(f"Number of unique images: {len(dataset['images'])}")
    print(f"Number of annotations: {len(dataset['annotations'])}")
    print(f"Average annotations per image: {len(dataset['annotations']) / len(dataset['images']):.2f}")
    
    return dataset['images'], dataset['annotations']

def main():
    # Set up paths
    base_dir = '/Users/aidanquigley/Documents/SAM2.small'
    output_dir = 'processed_data'
    
    # Process train dataset
    train_dir = os.path.join(base_dir, 'Train')
    print("\nStarting train data preparation...")
    train_images, train_annotations = process_dataset(train_dir, output_dir, 'train')
    
    # Process validate dataset
    validate_dir = os.path.join(base_dir, 'Validate')
    print("\nStarting validate data preparation...")
    validate_images, validate_annotations = process_dataset(validate_dir, output_dir, 'validate')
    
    print(f"\nProcessing complete!")
    print(f"Train dataset: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Validate dataset: {len(validate_images)} images, {len(validate_annotations)} annotations")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 