# 🫀 ECG Heartbeat Classification using HMM

A machine learning system for ECG heartbeat classification using **Hidden Markov Models (HMMs)** implemented from scratch.

---

## 📌 Project Idea

The system simulates ECG heartbeat signals, extracts temporal features, and classifies heartbeats using a custom Gaussian HMM trained per class.

---

## 🧠 Key Components

### 1. Synthetic ECG Generator
Generates heartbeat signals for 5 classes:
- Normal (N)
- Supraventricular (S)
- Ventricular (V)
- Fusion (F)
- Pacemaker (Q)

Includes realistic ECG behavior (PQRST shape, noise, baseline drift).

---

### 2. Feature Extraction
Each beat is converted into a time-series feature representation using:
- Amplitude
- Derivatives (1st & 2nd)
- Envelope
- Rolling energy
- Cumulative energy

---

### 3. Hidden Markov Model (from scratch)
Implemented using NumPy:
- Forward / Backward algorithms  
- Baum-Welch (training)  
- Viterbi decoding  
- Log-likelihood scoring  

---

### 4. Streamlit App
Interactive interface for:
- ECG visualization  
- Training HMM models  
- Classifying uploaded beats  
- Showing confidence scores  
- Evaluation results  

---

## 📂 Project Structure

```text
ecg-project/
│
├── ecg_data.py        # ECG signal generator
├── features.py        # Feature extraction
├── hmm_model.py       # HMM implementation
├── app.py             # Streamlit UI
├── make_beat.py       # Generate CSV test samples
│
├── sample_N.csv
├── sample_S.csv
├── sample_V.csv
├── sample_F.csv
└── sample_Q.csv

---

⚙️ How to Run

1. Generate ECG Samples
python make_beat.py

2. Run the Streamlit App
streamlit run app.py
