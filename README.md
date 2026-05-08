# 🫀 ECG Heartbeat Classification using Hidden Markov Models

A complete ECG heartbeat classification system built from scratch using Gaussian Hidden Markov Models (HMMs), synthetic ECG signal generation, temporal feature extraction, and an interactive Streamlit application.

---

# 📌 Overview

This project demonstrates how sequential probabilistic models can be applied to ECG heartbeat classification.

The system generates synthetic ECG heartbeat waveforms, extracts temporal features, trains one Gaussian HMM per class, and performs classification using sequence log-likelihood scoring.

The entire HMM implementation was developed from scratch using NumPy and SciPy.

---

# ✨ Features

## 🧠 Custom Gaussian HMM Implementation
- Forward Algorithm  
- Backward Algorithm  
- Baum-Welch (EM Training)  
- Viterbi Decoding  
- Log-Likelihood Scoring  

---

## 🫀 Synthetic ECG Generator
Generates realistic heartbeat signals for 5 classes:

- Normal Beat (N)  
- Supraventricular Beat (S)  
- Ventricular Beat / PVC (V)  
- Fusion Beat (F)  
- Pacemaker Beat (Q)  

Each signal includes:
- PQRST morphology simulation  
- Baseline wander  
- Additive noise  
- Random variations  

---

## 📊 Feature Extraction

Each heartbeat is converted into a temporal sequence using:

- Amplitude  
- First Derivative  
- Second Derivative  
- Envelope  
- Rolling Energy  
- Cumulative Energy  

---

## 🖥 Streamlit Application

The interactive app provides:

- ECG waveform visualization  
- Beat preprocessing visualization  
- HMM training and evaluation  
- CSV heartbeat classification  
- Confidence score visualization  
- Confusion matrix  
- Error analysis  

---

# 📂 Project Structure

```text
project/
│
├── ecg_data.py        # ECG synthetic generator
├── features.py        # Feature extraction pipeline
├── hmm_model.py       # Custom HMM implementation
├── make_beats.py      # Generate CSV demo samples
├── app.py             # Streamlit UI
│
├── sample_N.csv
├── sample_S.csv
├── sample_V.csv
├── sample_F.csv
└── sample_Q.csv
