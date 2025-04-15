# Heart Disease Prediction using Machine Learning

## Overview
This project aims to predict whether a person has heart disease (binary classification: 0/1) using different machine learning models. We evaluate and compare various models to determine the best one based on multiple evaluation metrics.

## Machine Learning Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree

## Dataset
The dataset used in this project is **Cardiovascular_Disease_Dataset**, collected from [Mendeley Data](https://data.mendeley.com/datasets/dzz48mvjht/1). It contains the following features:
The dataset contains the following features:
- `patientid`: Unique identifier for each patient (not used for training)
- `age`: Age of the patient
- `gender`: Gender of the patient
- `chestpain`: Type of chest pain experienced
- `restingBP`: Resting blood pressure
- `serumcholestrol`: Serum cholesterol level
- `fastingbloodsugar`: Fasting blood sugar level
- `restingrelectro`: Resting electrocardiographic results
- `maxheartrate`: Maximum heart rate achieved
- `exerciseangia`: Exercise-induced angina
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment
- `noofmajorvessels`: Number of major vessels colored by fluoroscopy
- `target`: 0 = No heart disease, 1 = Heart disease

## Evaluation Metrics
To compare the models, we use the following evaluation metrics:
- **Accuracy**: Measures overall correctness of predictions
- **Precision**: Measures the proportion of correctly predicted positive cases
- **Recall**: Measures how well the model identifies actual positive cases
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Measures the model's ability to distinguish between classes

## Implementation Steps
1. Load and preprocess the dataset
2. Encode categorical variables and normalize numerical features
3. Split the data into training and testing sets
4. Train and evaluate Logistic Regression, SVM, Random Forest, and Decision Tree models
5. Compare model performance based on evaluation metrics
6. Select the best-performing model

## Tools & Libraries Used
- **Python**
- **Google Colab**
- `pandas`, `numpy` (Data processing)
- `scikit-learn` (Machine Learning models and evaluation metrics)
- `matplotlib`, `seaborn` (Visualization)

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/AlImran1027/CSE299_Project
   ```
2. Open Google Colab and upload the dataset
3. Run the Jupyter Notebook step by step to train and evaluate models

## Results & Analysis
After evaluating all models using accuracy, precision, recall, F1-score, and AUC-ROC, we determine the best model for heart disease prediction.

## Future Improvements
- Feature selection to improve model performance
- Hyperparameter tuning for optimal results
- Exploring deep learning models

## Contributors
- Md.Mukzanul Alam Nishat
- Al Imran
- Arman Hossain Nawmee

## License
This project is licensed under the MIT License.

