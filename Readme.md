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

We developed a multi-model deep learning framework to classify coral reef health using RGB images. Our approach combines convolutional neural networks (CNNs) with a novel, ecologically sensitive loss function, enabling predictions that account not just for classification accuracy but also for the asymmetric severity of ecological misclassifications.

### 🧪 Pipeline Overview

1. **Data Preprocessing**  
   Raw images are normalized, resized to 128×128 pixels, and optionally enhanced via biologically motivated contrast and saturation adjustments. Labels are extracted and mapped to the ordinal class set: *dead*, *unhealthy*, *healthy*.

2. **Model Training**  
   We train three CNNs on the processed dataset:
   - **CNN₁**: A baseline architecture trained without augmentation.
   - **CNN₂ & CNN₃**: Identical architecture augmented with random rotation, horizontal/vertical flipping, and zooming to promote robustness to geometric and resolution variations.

   All models are trained using the **categorical cross-entropy (CCE)** loss and optimized via the Adam algorithm.

3. **Prediction Ensemble**  
   - Each model outputs class probabilities via a softmax layer.
   - These predictions are recalibrated through a post-training rescaling procedure that minimizes squared error between the predicted and true class indicators.
   - The final ensemble prediction is computed as a weighted sum of recalibrated outputs:  
     ```
     Ŷ_ensemble = ∑ₘ wₘ ⋅ Ŷ^{(m)*}
     ```
     where the Ŷ^{(m)*} are recalibrated model predictions and **w** is a learned convex weight vector.

4. **Loss Optimization**  
   - We define an **ecologically sensitive loss function** that penalizes misclassifications proportionally to their ecological severity. This loss is computed over the **confusion matrix** rather than raw prediction error.
   - To learn the optimal ensemble weights **w**, we minimize this non-differentiable loss using **simulated annealing** via `scipy.optimize.dual_annealing`, a global optimization routine well-suited for rugged objective landscapes.

---

This ensemble approach allows us to trade off between false positive types (e.g., misclassifying dead coral as healthy) by encoding environmental costs directly into the loss function.

After training, the best-performing model will be used to **automatically classify the Majuro dataset**, allowing us to create a new, labeled dataset for further research.

## 📌 Why This Matters  
Coral reef degradation is accelerating, and traditional monitoring methods are costly and time-intensive. By leveraging **machine learning and automation**, we can provide a **scalable, accurate, and cost-effective** tool for tracking coral reef health globally.

---

🔗 **Authors:** Nate Leary, Audrey Moessing, Sam Lee, Andrew Goldston, Aidan Quigley, w/ README file drafted by ChatGPT

📅 **Date:** April 2025  

📁 **Data Sources:** Hugging Face, Yellowfin Surfzone ASV (Majuro, Marshall Islands)  
