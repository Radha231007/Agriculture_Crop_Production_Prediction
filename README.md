# Agriculture Crop Production Prediction in India

## Company
UCT

## Project Overview

This project uses machine learning to predict agriculture crop production in India.

The project uses agriculture production data provided as part of the UCT Machine Learning Internship Projects.

The dataset contains crop production information for multiple crops across different agricultural years.

## Problem Statement

Agriculture production can vary depending on the crop and year.

The objective of this project is to develop a machine learning model that can learn patterns from historical crop production data and predict crop production.

## Dataset

The project uses the UCT-provided agriculture dataset.

The production dataset used for the baseline model contains:

- Crop
- Production 2006-07
- Production 2007-08
- Production 2008-09
- Production 2009-10
- Production 2010-11

The data was converted from wide format into a year-wise format.

## Data Preparation

The following steps were performed:

1. Loaded the CSV dataset.
2. Cleaned column names.
3. Converted production data from wide format to long format.
4. Created a Year column.
5. Removed missing values.
6. Encoded crop names using LabelEncoder.
7. Converted the year into a numerical feature.

The processed dataset is stored in:

`data/processed/agriculture_cleaned.csv`

## Machine Learning Model

A Random Forest Regressor was used for predicting crop production.

Features:

- Crop
- Year

Target:

- Production

The dataset was divided into training and testing sets using an 80:20 split.

## Model Results

| Metric | Result |
|---|---:|
| MAE | 29.7601 |
| RMSE | 38.7184 |
| R² | 0.5661 |

The R² score of 0.5661 indicates that the baseline model explains approximately 56.61% of the variation in the test data.

## Results Visualization

The project includes an Actual vs Predicted graph.

Location:

`results/figures/actual_vs_predicted.png`

## Project Structure

Agriculture_Crop_Production_Prediction/

- data/
  - raw/
  - processed/
- models/
- results/
  - figures/
- src/
- .gitignore
- README.md
- requirements.txt
- train_model.py

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- VS Code
- GitHub

## Learnings

Through this project, I learned:

- How to work with agriculture datasets.
- How to clean and transform data.
- How to convert wide-format data into long-format data.
- How to encode categorical variables.
- How to train a Random Forest regression model.
- How to evaluate a machine learning model.
- How to calculate MAE, RMSE and R².
- How to visualize actual and predicted values.
- How to organize a machine learning project.
- How to store and manage a project using GitHub.

## Conclusion

A Random Forest regression model was developed for agriculture crop production prediction.

The baseline model achieved an MAE of 29.7601, RMSE of 38.7184 and R² of 0.5661 on the test data.

This project demonstrates a complete machine learning workflow from data preparation to model training and evaluation.