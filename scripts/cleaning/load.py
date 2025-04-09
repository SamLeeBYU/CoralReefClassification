import os
from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class CoralHealthDataset:
    def __init__(self, data_dirs=["data/coral"]):
        """
        Initializes the dataset loader with the specified directory.
        """
        self.data_dir = data_dirs
        self.image_paths = self._get_image_paths()
        self.dataset = self._create_dataset()

    def _get_image_paths(self):
        """
        Retrieves all image file paths from the dataset directory.
        """
        image_extensions = (".png", ".jpg", ".jpeg")
        all_paths = []

        for data_dir in self.data_dir:
            if not os.path.isdir(data_dir):
                continue
            for fname in os.listdir(data_dir):
                if fname.lower().endswith(image_extensions):
                    all_paths.append(os.path.join(data_dir, fname))

        return all_paths

    @staticmethod
    def _extract_label(filename):
        """
        Extracts the label from the filename. Assumes format: 'label_xxx.png'
        """
        return filename.split("_")[0]

    def _load_image(self, file_path):
        """
        Loads an image using PIL and returns it.
        """
        return Image.open(file_path).convert("RGB") 
    
    def _create_dataset(self):
        """
        Reads images, extracts labels, and creates a Hugging Face dataset.
        """
        data = []
        for img_path in tqdm(self.image_paths, desc="Loading images"):
            label = self._extract_label(os.path.basename(img_path))
            image = self._load_image(img_path)
            data.append({"image": image, "label": label})

        return Dataset.from_list(data)

    def preview_data(self, num_samples=5):
        """
        Plots a random selection of sample images with their extracted labels.
        """

        random_indices = random.sample(range(len(self.dataset)), num_samples)

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

        for i, idx in enumerate(random_indices):
            example = self.dataset[idx]
            image = example["image"]
            label = example["label"]

            axes[i].imshow(image)
            axes[i].set_title(label)
            axes[i].axis("off")

        plt.show()

if __name__ == "__main__":
    coral = CoralHealthDataset(data_dir="data/coral")
    coral.preview_data()