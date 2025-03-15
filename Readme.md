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
We explore **three primary machine learning methods**:  

- **Support Vector Machines (SVMs)** – Effective for structured feature-based classification.  
- **Ordered Logistic Regression with Ridge Penalties** – Used for ordinal classification while preventing overfitting.  
- **Convolutional Neural Networks (CNNs)** – State-of-the-art deep learning models for image classification.  

After training, the best-performing model will be used to **automatically classify the Majuro dataset**, allowing us to create a new, labeled dataset for further research.

## 📌 Why This Matters  
Coral reef degradation is accelerating, and traditional monitoring methods are costly and time-intensive. By leveraging **machine learning and automation**, we can provide a **scalable, accurate, and cost-effective** tool for tracking coral reef health globally.

---
🔗 **Authors:** Nate Leary, Audrey Moessing, Sam Lee, Andrew Goldston, Aidan Quigley  
📅 **Date:** March 2025  
📁 **Data Sources:** Hugging Face, Yellowfin Surfzone ASV (Majuro, Marshall Islands)  
