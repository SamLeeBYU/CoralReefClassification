import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
from datasets import DatasetDict
from load import CoralHealthDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class CoralDataPreprocessor:
    def __init__(self, dataset_dict: DatasetDict, image_size=(128, 128), test_size=0.2, random_seed=42):
        """
        Initializes the preprocessor with the dataset and image size.
        """
        self.dataset = dataset_dict
        self.image_size = image_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.label_to_idx = self._create_label_mapping()

        self.X, self.y = self.process_dataset()
        self.train_test_split()

    def _create_label_mapping(self):
        """
        Creates a mapping from class labels to integer indices.
        """
        labels = list(set(example["label"] for example in self.dataset))
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def enhance_image(self, image):
        """
        Applies moderate contrast and saturation enhancement.
        """
        image = Image.fromarray(image)

        #Increase contrast
        contrast = ImageEnhance.Contrast(image)
        #1.0 means no change, > 1.0 increases contrast
        image = contrast.enhance(1) 

        #Increase saturation
        saturation = ImageEnhance.Color(image)
        image = saturation.enhance(1)

        return np.array(image)

    def extract_features(self, image):
        """
        Extracts CNN-compatible features by resizing and normalizing the image.
        """
        # Convert to OpenCV format (BGR) if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        #Resize image
        image = cv2.resize(image, self.image_size)

        #Normalize pixel values (0 to 1)
        image = image / 255.0

        return image

    def process_dataset(self):
        """
        Processes the dataset into NumPy matrices for CNN training.
        """
        X, y = [], []
        
        for example in tqdm(self.dataset, desc="Processing images"):
            image = np.array(example["image"])
            label = example["label"]

            #Enhance and preprocess image
            enhanced_image = self.enhance_image(image)
            cnn_features = self.extract_features(enhanced_image)

            #Store feature matrix and integer label
            X.append(cnn_features)
            y.append(self.label_to_idx[label])

        return np.array(X), np.array(y)
    
    def get_data(self):
        """
        Returns the full dataset stored within the object.
        """
        return self.X, self.y

    def train_test_split(self):
        """
        Splits dataset into train and test sets and stores them in the object.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_seed, stratify=self.y
        )

        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")

    def summary(self):
        dead = self.X[y == 0]
        unhealthy = self.X[y == 2]
        healthy = self.X[y == 1]

        print(f"Dead coral: {dead.shape[0]} samples")
        print(f"Unhealthy coral: {unhealthy.shape[0]} samples")
        print(f"Healthy coral: {healthy.shape[0]} samples")

        #Proportion of white pixels in each class
        #We classify white pixels as those with RGB values > (200, 200, 200)

        def calculate_white_proportion(image):
            #Check which pixels have all RGB values >= 200/255
            white_pixels = np.all(image >= 200/255, axis=-1)
            #Calculate the proportion of white pixels
            return np.mean(white_pixels)

        #Calculate white pixel proportions for each class
        dead_whites = [calculate_white_proportion(image) for image in dead]
        unhealthy_whites = [calculate_white_proportion(image) for image in unhealthy]
        healthy_whites = [calculate_white_proportion(image) for image in healthy]

        print(f"Dead coral: Mean proportion of white pixels: {np.mean(dead_whites):.4f}, Std: {np.std(dead_whites):.4f}")
        print(f"Unhealthy coral: Mean proportion of white pixels: {np.mean(unhealthy_whites):.4f}, Std: {np.std(unhealthy_whites):.4f}")
        print(f"Healthy coral: Mean proportion of white pixels: {np.mean(healthy_whites):.4f}, Std: {np.std(healthy_whites):.4f}")

        def calculate_channel_stats(image, channel_index):
            #Extract the specified channel (0: red, 1: green, 2: blue)
            channel = image[:, :, channel_index]
            #Calculate the mean and standard deviation of the channel
            mean_channel = np.mean(channel)
            std_channel = np.std(channel)
            return mean_channel, std_channel

        #Calculate red, green, and blue channel statistics for each class
        dead_reds = [calculate_channel_stats(image, 0) for image in dead]
        dead_greens = [calculate_channel_stats(image, 1) for image in dead]
        dead_blues = [calculate_channel_stats(image, 2) for image in dead]

        unhealthy_reds = [calculate_channel_stats(image, 0) for image in unhealthy]
        unhealthy_greens = [calculate_channel_stats(image, 1) for image in unhealthy]
        unhealthy_blues = [calculate_channel_stats(image, 2) for image in unhealthy]

        healthy_reds = [calculate_channel_stats(image, 0) for image in healthy]
        healthy_greens = [calculate_channel_stats(image, 1) for image in healthy]
        healthy_blues = [calculate_channel_stats(image, 2) for image in healthy]

        #Extract means and standard deviations for each class and channel
        dead_red_means, dead_red_stds = zip(*dead_reds)
        dead_green_means, dead_green_stds = zip(*dead_greens)
        dead_blue_means, dead_blue_stds = zip(*dead_blues)

        unhealthy_red_means, unhealthy_red_stds = zip(*unhealthy_reds)
        unhealthy_green_means, unhealthy_green_stds = zip(*unhealthy_greens)
        unhealthy_blue_means, unhealthy_blue_stds = zip(*unhealthy_blues)

        healthy_red_means, healthy_red_stds = zip(*healthy_reds)
        healthy_green_means, healthy_green_stds = zip(*healthy_greens)
        healthy_blue_means, healthy_blue_stds = zip(*healthy_blues)

        print("Dead Coral:")
        print(f"  Red - Mean: {np.mean(dead_red_means):.4f}, Std: {np.mean(dead_red_stds):.4f}")
        print(f"  Green - Mean: {np.mean(dead_green_means):.4f}, Std: {np.mean(dead_green_stds):.4f}")
        print(f"  Blue - Mean: {np.mean(dead_blue_means):.4f}, Std: {np.mean(dead_blue_stds):.4f}")

        print("Unhealthy Coral:")
        print(f"  Red - Mean: {np.mean(unhealthy_red_means):.4f}, Std: {np.mean(unhealthy_red_stds):.4f}")
        print(f"  Green - Mean: {np.mean(unhealthy_green_means):.4f}, Std: {np.mean(unhealthy_green_stds):.4f}")
        print(f"  Blue - Mean: {np.mean(unhealthy_blue_means):.4f}, Std: {np.mean(unhealthy_blue_stds):.4f}")

        print("Healthy Coral:")
        print(f"  Red - Mean: {np.mean(healthy_red_means):.4f}, Std: {np.mean(healthy_red_stds):.4f}")
        print(f"  Green - Mean: {np.mean(healthy_green_means):.4f}, Std: {np.mean(healthy_green_stds):.4f}")
        print(f"  Blue - Mean: {np.mean(healthy_blue_means):.4f}, Std: {np.mean(healthy_blue_stds):.4f}")

if __name__ == "__main__":
    coral = CoralHealthDataset(data_dir="data/coral")
    preprocessor = CoralDataPreprocessor(dataset_dict=coral.dataset)
    coral_processed = preprocessor.process_dataset()
    X, y = preprocessor.get_data()
    preprocessor.summary()

