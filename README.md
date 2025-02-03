# **Sentiment Classification in Internet Memes**

## **Overview**
This project focuses on **sentiment analysis in internet memes** using **text and image-based machine learning models**. The goal is to predict sentiment categories such as **humor, sarcasm, offensive, motivational, and overall sentiment** using a **multimodal deep learning approach**.

The project includes:
- **Model 1 (Text-Only Analysis)**: Uses **VADER sentiment analysis and RoBERTa-based transformer models** to classify meme text sentiment.
- **Model 2 (Multimodal Analysis - Image + Text)**: Combines **ResNet50 and VGG16 for image feature extraction** with **LSTM-based NLP models** for text sentiment analysis.

---

## **Dataset**
- **Source**: [Kaggle dataset](https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset) containing **6,992 meme images** and **text labels**.
- **Preprocessing Steps**:
  - **Text**: Stopword removal, punctuation cleanup, lemmatization, text embedding.
  - **Images**: Resized to **100×100 pixels**, normalized, and converted to RGB.

---

## **Model 1: Text-Based Sentiment Analysis**
- Utilizes **two approaches**:
  1. **VADER Sentiment Analysis**: Assigns sentiment scores (positive, negative, neutral) to meme text.
  2. **Transformer Model (RoBERTa)**:
     - Fine-tuned using **Twitter-RoBERTa** for meme-specific sentiment detection.
     - Achieved **~27% accuracy** due to complex language and sarcasm detection challenges.

---

## **Model 2: Multimodal Sentiment Analysis (Image + Text)**
- **Image Processing**:
  - Used **ResNet50 and VGG16** for feature extraction.
  - Pretrained weights from **ImageNet** applied for transfer learning.
- **Text Processing**:
  - Implemented **Bidirectional LSTM + Conv1D layers** for deep text embedding.
  - Converted meme text into **numerical sequences** for neural network input.
- **Final Model**:
  - **Concatenates image and text embeddings**.
  - Uses a **fully connected neural network** for final classification.
  - Achieved **~45% accuracy**, outperforming text-only models.

---

## **Technologies Used**
- **Programming Language**: Python
- **Deep Learning Libraries**: TensorFlow, PyTorch, Hugging Face Transformers
- **Data Processing**: Pandas, NumPy, OpenCV
- **NLP Models**: VADER, RoBERTa, LSTM, TextVectorization
- **Computer Vision Models**: ResNet50, VGG16

---

## **Contributors**
- Gandhar Ravindra Pansare
- Saish Vasudev Mhatre
- Tanmayee Tajane
