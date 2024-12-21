# README: Emotion Recognition using EmoDB Dataset
## Overview

This project implements an emotion recognition system using the EmoDB dataset, a collection of labeled audio files representing various emotional states. The system extracts features from the audio files, trains machine learning classifiers, and evaluates their performance in predicting emotions such as Angry, Happy, Sad, and Neutral.
Features Extracted

- MFCC (Mel-Frequency Cepstral Coefficients): Represents the power spectrum of the audio signal.
- Chroma Features: Captures the tonal content and pitch information.
- Mel Spectrogram: Encodes the short-term power spectrum in the Mel scale.
- STFT (Short-Time Fourier Transform): Calculates the frequency spectrum over time.

## Machine Learning Models

The following classifiers are trained and evaluated:

- Decision Tree Classifier
- Support Vector Machines (SVM)
- Random Forest Classifier
- Naive Bayes Classifier
- K-Nearest Neighbors (KNN)

## Key Libraries

- Audio Processing: librosa, soundfile
- Machine Learning Models: sklearn
- Visualization: matplotlib, seaborn
- Data Handling: numpy, pandas

# Workflow
1. Data Preprocessing

    Audio features are extracted from .wav files using the extract_feature function.
    Features and labels are split into training and testing sets using train_test_split.

2. Model Training and Evaluation

    Each model is trained using the training set.
    Models are evaluated using metrics such as:
        Accuracy
        Precision
        Recall
        F1-Score
        AUC-ROC
    Confusion matrices and ROC curves are visualized for better understanding.

3. Accuracy Comparison

    The performance of all models is compared using a bar chart.
    Precision, recall, and F1-scores are also plotted for detailed comparison.

## Results

Each model's performance is measured and compared:

- Decision Tree: Competitive performance with interpretable results.
- SVM: High accuracy and robust generalization.
- Random Forest: Reliable and achieves high scores across metrics.
- Naive Bayes: Simple but effective for smaller datasets.
- KNN: Performs well when features are scaled but lacks predict_proba by default.

## Prerequisites

Install required Python libraries:

```pip install librosa soundfile numpy scikit-learn matplotlib seaborn pandas```

Place the EmoDB .wav files in the designated directory:

```/content/drive/MyDrive/EmoDB/```


Visualization Examples

- Accuracy Comparison: A bar chart comparing the accuracy of all classifiers.
- Confusion Matrices: Heatmaps showing the distribution of predicted vs. true labels for each classifier.
- ROC Curves: Multi-class ROC curves plotted for supported classifiers.

Future Improvements

- Add support for additional emotions.
- Experiment with deep learning models for improved accuracy.
- Fine-tune hyperparameters using techniques like grid search.
- Incorporate larger and more diverse datasets.