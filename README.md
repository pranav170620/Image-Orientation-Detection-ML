# Image Orientation Detection with KNN & PCA

This project demonstrates the application of scalable machine learning techniques to real-world datasets, tackling multiple predictive and analytical tasks using PySpark.
The workflow is divided into modular stages — data preparation, model training, and evaluation — allowing for flexible experimentation with different modeling approaches. Three joblib-serialized models trained with varying hyperparameters are stored and compared.

### Key components include:

Data Preparation: Raw data is cleaned and engineered using prepare_data.py to optimize feature quality and handle categorical variables and outliers.
Model Training: train_model.py trains and stores multiple models with iteration-controlled configurations (model.30.joblib, model.50.joblib, model.90.joblib).
Model Evaluation: evaluate.py compares model performance using classification metrics such as accuracy, AUC, and precision-recall, and visualizes misclassification patterns.
This project showcases proficiency in modular Python design, model serialization, hyperparameter tuning, and evaluation strategies — essential skills for modern data analysts and applied ML engineers.

### What It Does

- Processes facial image patches of sizes **30x30**, **50x50**, and **90x90**
- Applies PCA to compress features
- Trains separate KNN models for each size
- Evaluates performance on test sets

### Accuracy

| Resolution | Accuracy |
|------------|----------|
| 90x90      | 90.7%    |
| 50x50      | 59.85%   |
| 30x30      | 33.65%   |

### ▶️ How to Run

```bash
python prepare_data.py
python train_model.py
python evaluate.py model.90.joblib 90
