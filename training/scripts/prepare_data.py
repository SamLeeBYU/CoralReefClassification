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
    
    # Check if this is a dataset without annotations
    jsons_dir = os.path.join(data_dir, 'jsons')
    if not os.path.exists(jsons_dir):
        print(f"\nProcessing {dataset_type} dataset (no annotations)...")
        # Just copy images
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in tqdm(image_files, desc=f"Processing {dataset_type} images"):
            src_img_path = os.path.join(data_dir, img_file)
            dst_img_path = os.path.join(output_images_dir, img_file)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Add image info to dataset
            img = cv2.imread(src_img_path)
            if img is not None:
                height, width = img.shape[:2]
                image_info = {
                    'file_name': img_file,
                    'height': height,
                    'width': width,
                    'id': len(dataset['images'])
                }
                dataset['images'].append(image_info)
        
        # Save empty annotations file
        output_json = os.path.join(output_annotations_dir, 'dataset.json')
        with open(output_json, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return dataset['images'], dataset['annotations']
    
    # Process dataset with annotations
    print(f"\nProcessing {dataset_type} dataset...")
    json_files = [f for f in os.listdir(jsons_dir) if f.endswith('.json')]
    
    # First, create a mapping from image filename to new image ID
    filename_to_id = {}
    current_img_id = 0
    current_ann_id = 0
    
    for json_file in tqdm(json_files, desc=f"Processing {dataset_type} files"):
        json_path = os.path.join(jsons_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file: {json_file}")
            continue
        
        # Validate image info
        if not validate_image_info(data['image']):
            print(f"Warning: Invalid image info in {json_file}")
            continue
        
        img_name = data['image']['file_name']
        
        # Skip if we've already processed this image
        if img_name in filename_to_id:
            continue
            
        # Copy image to output directory
        src_img_path = os.path.join(data_dir, 'images', img_name)
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
        json.dump(dataset, f, indent=2)
    
    return dataset['images'], dataset['annotations']

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'processed_data')
    
    # Process each dataset
    for dataset_type in ['train', 'validate', 'test']:
        dataset_path = os.path.join(data_dir, dataset_type)
        if not os.path.exists(dataset_path):
            print(f"\nSkipping {dataset_type} dataset - directory not found at {dataset_path}")
            continue
            
        try:
            images, annotations = process_dataset(dataset_path, output_dir, dataset_type)
            print(f"\n{dataset_type.capitalize()} Dataset Summary:")
            print(f"Total images: {len(images)}")
            print(f"Total annotations: {len(annotations)}")
            if len(images) > 0:
                print(f"Average annotations per image: {len(annotations)/len(images):.2f}")
        except Exception as e:
            print(f"\nError processing {dataset_type} dataset: {str(e)}")
            continue

if __name__ == "__main__":
    main() 