🫀 ECG Heartbeat Classification using Hidden Markov Models

A complete ECG heartbeat classification system built from scratch using Gaussian Hidden Markov Models (HMMs), synthetic ECG signal generation, temporal feature extraction, and an interactive Streamlit application.

📌 Overview

This project demonstrates how sequential probabilistic models can be applied to ECG heartbeat classification.

The system generates synthetic ECG heartbeat waveforms inspired by the MIT-BIH Arrhythmia Database, extracts temporal features, trains one Gaussian HMM per heartbeat class, and performs classification using sequence log-likelihood scoring.

The entire HMM implementation was developed from scratch using NumPy and SciPy.

✨ Features
🧠 Custom Gaussian HMM Implementation

Implemented from scratch:

Forward Algorithm
Backward Algorithm
Baum-Welch (EM) Training
Viterbi Decoding
Log-Likelihood Scoring
🫀 Synthetic ECG Generator

The project includes a realistic heartbeat generator capable of producing:

Normal Beat (N)
Supraventricular Beat (S)
Ventricular Beat / PVC (V)
Fusion Beat (F)
Pacemaker Beat (Q)

Signals include:

PQRST morphology modeling
Baseline wander
Additive Gaussian noise
Randomized waveform variability
📊 Temporal Feature Extraction

Each ECG beat is converted into a sequential observation matrix using:

Feature	Description
Amplitude	Normalized ECG amplitude
1st Derivative	Local slope information
2nd Derivative	Curvature / peak detection
Envelope	Absolute signal amplitude
Rolling Energy	Local signal energy
Cumulative Energy	Energy accumulation trend
🖥 Interactive Streamlit Interface

The application provides:

ECG visualization
Beat preprocessing visualization
HMM training interface
Real-time classification
CSV heartbeat upload
Evaluation metrics
Confusion matrix
Error analysis
EM convergence visualization
📂 Project Structure
project/
│
├── ecg_data.py          # Synthetic ECG generator
├── features.py          # Feature extraction pipeline
├── hmm_model.py         # Custom Gaussian HMM implementation
├── make_beats.py        # Generate demo ECG CSV files
├── app.py               # Streamlit application
│
├── record_01.csv
├── record_02.csv
├── record_03.csv
├── record_04.csv
└── record_05.csv

▶️ Running the Project
Step 1 — Generate ECG Demo Samples
python make_beats.py

This creates sample ECG heartbeat CSV files for testing the classifier.

Step 2 — Launch Streamlit App
streamlit run app.py
🧪 Classification Workflow

The application supports:

Uploading ECG beat CSV files
Generating synthetic heartbeat samples
Visualizing waveform morphology
Predicting heartbeat class
Viewing confidence scores
📈 Example Classes
Label	Description
N	Normal Beat
S	Supraventricular Beat
V	Ventricular Beat / PVC
F	Fusion Beat
Q	Pacemaker Beat
📚 Educational Purpose

This project is intended for:

Pattern Recognition learning
Sequential probabilistic modeling
ECG signal processing experiments
HMM implementation practice
Machine Learning portfolio projects
🛠 Technologies
Python
NumPy
SciPy
Scikit-learn
Matplotlib
Streamlit
📌 Future Improvements

Potential future extensions:

Real MIT-BIH ECG integration
Real-time ECG streaming
Deep learning comparison (LSTM / CNN)
Advanced signal denoising
Model deployment
Attention-based sequence models
👩‍💻 Author

Developed as a Pattern Recognition and Biomedical Signal Processing project.
