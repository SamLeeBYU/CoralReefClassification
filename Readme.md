# 🌊 Regrading Degrading: Predicting the Current Degradation of Coral Reefs 

## 📌 Project Overview  
This project aims to develop a **machine learning-based classification model** to assess coral reef health using image data. Coral reefs play a critical role in marine biodiversity, coastal protection, and global fisheries. However, they are under increasing threat due to **ocean acidification, global warming, destructive fishing practices, and pollution**. Our goal is to build an **automated system** that can classify coral reefs as **healthy, partially bleached, or fully bleached**, providing an efficient alternative to manual reef monitoring.

Our research seeks to **train machine learning models on existing labeled datasets** and use these models to **generate new labeled datasets from previously unlabeled coral reef images** collected in the Majuro, Marshall Islands.

## 📊 Data Sources  
We use **two main datasets**:

1. **Hugging Face Coral Health Classification Dataset**  
   - A publicly available dataset of labeled coral images.  
   - Contains images classified as **healthy, unhealthy, and dead**.  
   - Used for **training our models**.  

2. **Yellowfin Surfzone ASV Dataset (Majuro, Marshall Islands)**  
   - Consists of **unlabeled coral reef images** collected from Majuro.  
   - Our trained model will **generate new labeled data** from this dataset.  

Each image is preprocessed to extract key features, including **RGB pixel values, white pixel proportions, and texture-based features**.

## 🤖 Machine Learning Approach  

We implemented a multi-model deep learning framework to classify coral health using RGB images. Our pipeline consists of:

- **Three CNNs** trained on preprocessed coral reef images, with two models utilizing random data augmentation.
- Each model is trained using **categorical cross-entropy loss**.
- Final predictions are made via a **weighted ensemble** over softmax probabilities.

### 🧪 Pipeline Overview

1. **Data Preprocessing**  
   Includes normalization, resizing, and optional image enhancements.

2. **Model Training**
   - **CNN₁**: Base architecture (no augmentation)  
   - **CNN₂ & CNN₃**: Same architecture with random rotation, flipping, and zoom

3. **Prediction Ensemble**
   - Ensemble probabilities are computed as:  
     \[
     \hat{Y}_{\text{ensemble}} = \sum_m \gamma_m \, \hat{Y}^{(m)}
     \]
   - Final predictions are taken as:  
     \[
     \hat{y}_i = \arg\max_j \hat{Y}_{ij}
     \]

4. **Loss Optimization**
   - The $\gamma$ weights are learned by minimizing an **ecologically sensitive loss function**  
     defined over the confusion matrix, not raw prediction error.
   - Optimization is performed via **simulated annealing** using  
     `scipy.optimize.dual_annealing`, which is robust to non-smooth, non-convex objectives.

---

This ensemble approach allows us to trade off between false positive types (e.g., misclassifying dead coral as healthy) by encoding environmental costs directly into the loss function.

After training, the best-performing model will be used to **automatically classify the Majuro dataset**, allowing us to create a new, labeled dataset for further research.

## 📌 Why This Matters  
Coral reef degradation is accelerating, and traditional monitoring methods are costly and time-intensive. By leveraging **machine learning and automation**, we can provide a **scalable, accurate, and cost-effective** tool for tracking coral reef health globally.

---
🔗 **Authors:** Nate Leary, Audrey Moessing, Sam Lee, Andrew Goldston, Aidan Quigley  
📅 **Date:** March 2025  
📁 **Data Sources:** Hugging Face, Yellowfin Surfzone ASV (Majuro, Marshall Islands)  
