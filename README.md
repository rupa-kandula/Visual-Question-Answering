# Visual-Question-Answering

# 🤖 Visual Question Answering (VQA) using CLIP + ResNet Framework

This repository contains the implementation and results of our VQA research:

> **Bridging Vision and Language: A CLIP-ResNet Framework for Visual Question Answering**  
> 📍 Authors: Rupa Kandula, Jishnu Teja Dandamudi, Rama Muni Reddy Yanamala  
> 🎓 Affiliation: Amrita Vishwa Vidyapeetham, Coimbatore, India  
> 🗓️ Year: 2025  
> 📄 Dataset: DAQUAR (publicly available)

---

## 🔍 Abstract

Visual Question Answering (VQA) is a multimodal AI task that requires understanding both images and natural language to answer visual questions. In this project, we propose a **hybrid model** that fuses:

- **Visual features** from a pre-trained **ResNet-18**
- **Textual embeddings** from **CLIP (Contrastive Language-Image Pretraining)**

These features are concatenated and passed through a classification layer to predict the correct answer from a predefined set.

---

## 🧠 Model Architecture

- 🔷 **Image Encoder**: ResNet-18 (output: 512-dim vector)
- 🔶 **Text Encoder**: CLIP tokenizer and text transformer (output: 512-dim vector)
- 🔗 **Fusion Layer**: Concatenation → 1024-dim joint vector
- 🎯 **Classifier**: Fully Connected → Softmax → Answer Prediction

### 📊 Training Configuration

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Evaluation Metrics: Accuracy, Weighted Precision
- Best Accuracy Achieved: **95.4%** on DAQUAR dataset

---


