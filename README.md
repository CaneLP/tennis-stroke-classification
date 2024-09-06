# Tennis Stroke Classification Project 🎾

This project focuses on classifying tennis strokes using a range of machine learning and deep learning models. The dataset consists of short video clips, each lasting 1 second, which have been manually relabeled to enhance model performance. Initially, the videos were obtained through an automated pipeline employing machine learning models, though the specifics of that process are beyond the scope of this project. The primary objective here is to evaluate different models to determine the most effective approach for tennis stroke classification.

## Table of Contents 📃

- [Project Overview](#project-overview-)
- [Dataset](#dataset-)
- [Modeling Techniques and Evaluations](#modeling-techniques-and-evaluations-)
- [Results](#results-)
- [Installation and Usage](#installation-and-usage-)
- [Important Tips](#important-tips-)

## Project Overview 🔎

This project implements multiple models, including traditional machine learning techniques like Logistic Regression and SVM, as well as deep learning methods such as MLP or CNNs, to classify tennis strokes. The goal is to compare these methods on both initial and cleaned datasets, assessing their performance in terms of accuracy, F1 score, and other relevant metrics.

## Dataset 🗃

- **Size:** ~12,000 short video clips (1 second each)
- **Relabeling Process:** To make sure the dataset was accurate, we went through all 12,850 video clips from 27 Grand Slam matches and fixed any errors we found. This process took about 20 active hours spread over a few weeks (around 9 minutes for every 100 strokes). It was an important step to build a good dataset for this project. As Andrew Ng mentioned, focusing on data quality is key for AI success ([Fortune article](https://fortune.com/2022/06/21/andrew-ng-data-centric-ai/)).
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

  This process of turning videos into 'action images' is based on the approach described in [this paper](https://arxiv.org/pdf/1704.05645.pdf). Note that the red and green channels in the image represent the x and y coordinates of the keypoints, while the blue channel represents the confidence level of those keypoints.

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

Additionally, we developed a more complex CNN model with extra layers and dropout, training it for 100 epochs. This advanced model showed more consistent results across the training, validation, and test sets, indicating better generalization compared to the initial model, which was also trained for 100 epochs. However, all the CNN models produced fairly similar overall results.

## Results 🚀

Results are only shown for the test set. For a more detailed version, including training and validation results, as well as confusion matrices, please refer to the notebooks.

| Model                   | Dataset  | Accuracy | Precision | Recall | F1 Score (macro) |
|--------------------------|----------|----------|-----------|--------|------------------|
| **Logistic Regression**  | Initial  | 0.78     | 0.77      | 0.75   | 0.76             |
| **Logistic Regression**  | Cleaned  | 0.91     | 0.90      | 0.88   | 0.89             |
| **SVM**                  | Initial  | 0.84     | 0.83      | 0.82   | 0.82             |
| **SVM**                  | Cleaned  | 0.94     | 0.93      | 0.92   | 0.92             |
| **Boosting**             | Initial  | 0.83     | 0.83      | 0.80   | 0.81             |
| **MLP**                  | Initial  | 0.82     | 0.82      | 0.79   | 0.80             |
| **MLP**                  | Cleaned  | 0.92     | 0.90      | 0.90   | 0.90             |
| **CNN (50 epochs)**      | Initial  | 0.85     | 0.85      | 0.84   | 0.85             |
| **CNN (50 epochs)**      | Cleaned  | 0.96     | 0.96      | 0.96   | 0.96             |
| **CNN (100 epochs)**     | Cleaned  | 0.96     | 0.96      | 0.96   | 0.96             |
| **CNN (Complex, 100 e)** | Cleaned  | **0.97** | **0.96**  | **0.96** | **0.96**       |

## Installation and Usage 🛠

To get started, follow the steps below to set up your environment and run the necessary scripts.

### 1. Clone the repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/CaneLP/tennis-stroke-classification.git
cd tennis-stroke-classification
```

### 2. Setup the environment with pipenv

Next, set up the Python 3.8 environment using `pipenv`:

```bash
pipenv --python 3.8
```

**Note:** While other Python versions `>3.8` should work, this project is developed and tested with 3.8.

After the environment is created, install all dependencies specified in the `Pipfile`:

```bash
pipenv install
```

**Installation may take around 20-25 minutes and requires approximately 9GB of memory due to the extensive dependencies of the Ultralytics library.**

### 3. Enter the pipenv shell

Activate the environment by entering the Pipenv shell:

```bash
pipenv shell
```

### 4. Download and prepare the dataset, sample files and models

Run the `download_dataset.py` script to download and unpack the necessary datasets. Note that this process may take a few minutes:

```bash
python download_dataset.py
```

Once the dataset is prepared, you can start using the Jupyter notebooks located inside the `notebooks/03_model_dev` directory:

If you want to explore some sample data, you can run the `download_samples.py` script:

```bash
python download_samples.py
```

After running this script, you can open and interact with the notebooks in the `notebooks/02_dataset_creation` directory. Note that running these notebooks will not actually create the full dataset, as that process is time-consuming and not practical for quick experiments. This is provided just for fun and exploration.

To download the pre-trained models, run the `download_models.py` script:

```bash
python download_models.py
```
2. **Start Jupyter Notebook**:

To use the correct environment in Jupyter notebook:

  **Add the Kernel**:
   ```bash
   pipenv shell  # if not already inside the environment
   python -m ipykernel install --user --name=tennisai --display-name "Tennis AI"
   ```

   ```bash
   jupyter notebook
   ```

  **Select the Kernel**:
   - In Jupyter, go to `Kernel > Change Kernel` and choose "Tennis AI".

## Important Tips 💡

1. **Data sampling for faster testing**: In each notebook (except for those using CNNs), there's a specific cell dedicated to data loading. You can adjust the number of examples to speed up model testing. Even with just 500 examples, you can achieve meaningful results. Simply uncomment and run the provided cell with your desired `num_examples` value. **Note:** After running one sampling, you must reload the data from the start to ensure proper functionality.

```python
# Uncomment and run this cell with the desired num_examples count
# if you'd like to work with less data for testing purposes

num_examples = 500
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=num_examples, stratify=y, random_state=42)
X = X_sampled
y = y_sampled

bh_cnt = sum([1 for l in y if l == 'backhand'])
fh_cnt = sum([1 for l in y if l == 'forehand'])
other_cnt = sum([1 for l in y if l == 'other'])
serve_cnt = sum([1 for l in y if l == 'serve'])
print(f'Backhands count: {bh_cnt}, Forehands count: {fh_cnt}, Other count: {other_cnt}, Serve count: {serve_cnt}')
print(f'All strokes count: {len(y)}')
```

2. **Optimizing processing speed with `n_jobs`**: The `n_jobs` parameter is currently commented out, but you can uncomment it to speed up computations. Set it to `-1` to utilize all available processing power, or experiment with specific values. Start with lower numbers, as setting it too high (e.g., 20+) can overwhelm your computer.
