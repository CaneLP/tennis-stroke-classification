# Tennis Stroke Classification Project 🎾

This project focuses on classifying tennis strokes using a range of machine learning and deep learning models. The dataset consists of short video clips, each lasting 1 second, which have been manually relabeled to enhance model performance. Initially, the videos were obtained through an automated pipeline employing machine learning models, though the specifics of that process are beyond the scope of this project. The primary objective here is to evaluate different models to determine the most effective approach for tennis stroke classification.

## Table of Contents 📃

- [Project Overview](#project-overview-)
- [Dataset](#dataset-)
- [Modeling Techniques and Evaluations](#modeling-techniques-and-evaluations-)
- [Results](#results-)
- [Installation and Usage](#installation-and-usage-)

## Project Overview 🔎

This project implements multiple models, including traditional machine learning techniques like Logistic Regression and SVM, as well as deep learning methods such as MLP or CNNs, to classify tennis strokes. The goal is to compare these methods on both initial and cleaned datasets, assessing their performance in terms of accuracy, F1 score, and other relevant metrics.

## Dataset 🗃

- **Size:** ~12,000 short video clips (1 second each)
- **Classes:** Four types of tennis strokes: forehand, backhand, serve, and an "other" category representing any player behavior not classified as one of the three primary strokes
- **Structure:** Each video originally contains approximately 23-30 frames, depending on the FPS, but all videos have been resampled to 20 frames for consistency.
- **Transformation:** These 1-second-long videos were transformed into 20x12 images with 3 channels (RGB) using pose estimation methods. Specifically, Ultralyics' model was utilized ([Ultralytics GitHub](https://github.com/ultralytics/ultralytics)). In this transformation:
  - **20:** Refers to the 20 frames on which the pose was tracked.
  - **12:** Refers to the 12 pose key points identified as relevant:
    - 5: Left Shoulder
    - 6: Right Shoulder
    - 7: Left Elbow
    - 8: Right Elbow
    - 9: Left Wrist
    - 10: Right Wrist
    - 11: Left Hip
    - 12: Right Hip
    - 13: Left Knee
    - 14: Right Knee
    - 15: Left Ankle
    - 16: Right Ankle

  This process of turning videos into 'action images' is based on the approach described in [this paper](https://arxiv.org/pdf/1704.05645.pdf).

## Modeling Techniques and Evaluations 🤖

The results, full process, and detailed analysis of the models are available in the appropriate notebooks located in the `notebooks/03_model_dev` folder.

- **Logistic Regression**
  - **Datasets:** Initial, Cleaned
  - **Hyperparameters:** Ran GridSearchCV for regularization methods (`l1`, `l2`, `elasticnet`).

- **Support Vector Machine (SVM)**
  - **Datasets:** Initial, Cleaned
  - **Hyperparameters:** Ran GridSearchCV for kernel type, regularization parameter (`C`), and kernel coefficient (`gamma`).

- **Boosting**
  - **Dataset:** Initial only
  - **Description:** Applied Boosting techniques to check how it behaves on this classification task.
  - **Hyperparameters:** Ran GridSearchCV for number of estimators, learning rate, and max depth.

- **Multilayer Perceptron (MLP)**
  - **Datasets:** Initial, Cleaned
  - **Hyperparameters:** Ran GridSearchCV for a few parameters, such as units per layer, activation function, learning rate, and similar.

- **Convolutional Neural Network (CNN)**
  - **Datasets:** Initial, Cleaned
  - **Description:** The CNN model did a great job at generalizing, which means it worked well not only on the data it was trained on but also on new, unseen data. The results on the test set were very close to those on the training and validation sets, showing that the model didn't overfit and could make accurate predictions on new data. This was different from the other models, which, even though they did well during training and validation, often performed worse on the test set. While those models still gave good results, the CNN was especially impressive because it kept its performance steady across all datasets. This made the CNN a very reliable choice for this classification task.

Additionally, we developed a more complex CNN model with some extra layers and dropout, and trained it for 200 epochs. This more advanced model gave the most consistent results across the training, validation, and test sets, showing it could generalize maybe even better than the initial model (also trained on 200 epochs).

## Results 🚀

Results are only shown for the test set. For a more detailed version, including training and validation results, as well as confusion matrices, please refer to the notebooks.

| Model                   | Dataset  | Accuracy | Precision | Recall | F1 Score (macro) |
|--------------------------|----------|----------|-----------|--------|------------------|
| **Logistic Regression**  | Initial  | 0.78     | 0.76      | 0.74   | 0.75             |
| **Logistic Regression**  | Cleaned  | 0.90     | 0.89      | 0.88   | 0.89             |
| **SVM**                  | Initial  | 0.83     | 0.82      | 0.81   | 0.81             |
| **SVM**                  | Cleaned  | 0.95     | 0.94      | 0.94   | 0.94             |
| **Boosting**             | Initial  | 0.82     | 0.81      | 0.79   | 0.80             |
| **MLP**                  | Initial  | 0.81     | 0.79      | 0.79   | 0.79             |
| **MLP**                  | Cleaned  | 0.95     | 0.94      | 0.93   | 0.93             |
| **CNN (50 epochs)**      | Initial  | 0.85     | 0.85      | 0.84   | 0.84             |
| **CNN (50 epochs)**      | Cleaned  | 0.96     | 0.95      | 0.96   | 0.96             |
| **CNN (200 epochs)**     | Cleaned  | 0.97     | 0.96      | 0.97   | 0.96             |
| **CNN (Complex, 200 e)** | Cleaned  | **0.97** | **0.97**  | **0.97** | **0.97**       |

## Installation and Usage 🛠

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/tennis-stroke-classification.git
cd tennis-stroke-classification
pip install -r requirements.txt # TODO
```
