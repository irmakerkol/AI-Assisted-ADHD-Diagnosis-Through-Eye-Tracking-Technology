# AI-Assisted ADHD Diagnosis Through Eye Tracking Technology

This project explores the use of **machine learning** to support the diagnosis of **Attention Deficit Hyperactivity Disorder (ADHD)** using only **eye-tracking data**. The system analyzes eye movement metrics—such as saccadic behavior, fixation stability, and intrusive saccades—collected from ADHD and healthy participants during visual attention tasks.  

## 🧠 Overview

- **Objective:** Determine whether machine learning models can distinguish ADHD patients from healthy individuals using features extracted from eye movements.
- **Approach:** Collect eye-tracking data during tasks, extract behavioral features, train multiple classifiers, and evaluate their performance.

## 📊 Dataset

- **Participants:**  
  - 8 ADHD-diagnosed individuals (confirmed by a psychiatrist)  
  - 9 healthy controls  
- **Device:** Tobii Pro X2-60 Eye Tracker (30Hz sampling rate)  
- **Tasks Implemented in Unity:**
  - **Fixation Task** – to measure intrusive saccades
  - **Prosaccade/Antisaccade Tasks** – to measure response inhibition and direction errors

## 🧮 Features Extracted

- **Mean Simple Reaction Time (SRT):** Average delay between stimulus appearance and the first saccade.
- **Coefficient of Variation of SRT:** Measures consistency of response times.
- **Number of Direction Errors:** Incorrect initial saccade direction during antisaccade trials.
- **Number of Intrusive Saccades:** Unintended saccades during fixation.

Saccade and intrusive thresholds were varied systematically (e.g., 0.5°–5.0°) during training to find optimal detection parameters.

## 🤖 Machine Learning Pipeline

- **Models Used:**
  - Support Vector Machines (SVM) with linear, polynomial, and RBF kernels
  - Logistic Regression
  - Random Forest Classifiers (with 20 and 100 estimators)

- **Evaluation Strategy:**
  - 5-fold Cross-Validation repeated **100 times**
  - **F1-macro** as performance metric

- **Best Performance:**
  - **SVM (RBF Kernel):**  
    - Mean F1-score: **0.77**  
    - Std(F1-score): **0.04**

## 🧪 Tools & Libraries

- `Python` for preprocessing and ML modeling
- `Pandas`, `NumPy` for data handling
- `scikit-learn` for ML models
- `Tobii Pro Lab` for eye-tracking
- `Unity` for task design and stimuli presentation

## 📈 Results

- **Eye-tracking data alone** enabled effective classification of ADHD vs. healthy individuals.
- The system consistently performed best with the SVM RBF model across different angle thresholds.
- Variability in results was low, indicating consistent feature quality and model stability.

## 🚧 Limitations

- Small dataset (n=17) may limit generalizability.
- Underfitting may be present; additional samples are needed for better tuning.

## 📌 Future Work

- Integrate more complex deep learning models (e.g., LSTM for temporal analysis).
- Test with larger and more diverse datasets.
- Explore real-time applications in clinical environments.

