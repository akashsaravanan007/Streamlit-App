# LLM-Based Synthetic Text Generation for Sentiment Analysis and Spam Detection

## Overview
This project presents an end-to-end system that uses a Large Language Model (LLM) to generate synthetic text data and applies machine learning techniques for classification tasks such as sentiment analysis and spam detection.

The system integrates prompt-based data generation, dataset construction, feature extraction, model training, and performance evaluation within a single interactive application built using Streamlit.

---

## Features
- Prompt-based synthetic text generation using Flan-T5
- Automatic dataset creation with labels
- Real-time data generation with progress tracking
- Dataset preview and CSV download option
- Text feature extraction using TF-IDF
- Classification using Logistic Regression
- Model evaluation with accuracy and confusion matrix
- Interactive web interface using Streamlit

---

##  Technologies Used
- Python
- Streamlit
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## Project Structure
├── app.py                # Main Streamlit application  
├── synthetic_datasets/   # Saved datasets (CSV files)  
├── README.md             # Project documentation  

---

##  Installation

1. Clone the repository:
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  

2. Install dependencies:
pip install -r requirements.txt  

If requirements.txt is not available:
pip install streamlit pandas numpy matplotlib scikit-learn transformers  

---

##  Usage

Run the Streamlit app:
streamlit run app.py  

---

##  How It Works

1. Enter Prompt  
2. Generate Data using Flan-T5  
3. Dataset Creation  
4. Train Model using TF-IDF + Logistic Regression  
5. Evaluate Model  

---

## Example Use Cases
- Sentiment Analysis  
- Spam Detection  
- Dataset generation for NLP tasks  
- Educational ML demonstrations  

---

##  Future Improvements
- Advanced LLM integration  
- More ML models  
- Multilingual support  
- Better visualization  

---------------------------------------------------------------------------------------------------
