import os
from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json

class CoralHealthDataset:
    def __init__(self, data_dirs=["data/coral"], image_size=(128, 128), annotations=None, annotations_dict = {
        #For the WHOI labels
        "coral": "healthy",
        "coral_bleach": "unhealthy"
    }):
        """
        Initializes the dataset loader with the specified directory.
        """
        self.data_dir = data_dirs
        self.annotations = annotations
        self.image_size = image_size

        if self.annotations is not None:
            with open(annotations, "r") as f:
                label_data = json.load(f)
            
                self.labels = {os.path.basename(k): v for k, v in label_data.items()}
                #Change the labels to the annotations specified in the dictionary
                for k, v in self.labels.items():
                    if v in annotations_dict:
                        self.labels[k] = annotations_dict[v]
                    else:
                        self.labels[k] = "unknown"

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

    def _extract_label(self, filename):
        """
        Extracts the label from the filename. Assumes format: 'label_xxx.png'
        """
        if self.annotations is None:
            return filename.split("_")[0]
        else:
            #Get the label from the filename key in the json file
            return self.labels.get(filename, "unknown")

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
            #load in image and resize it
            image = self._load_image(img_path).resize(self.image_size)
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
    coral = CoralHealthDataset(data_dirs="data/coral")
    coral.preview_data()