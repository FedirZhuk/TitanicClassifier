Simple exercise from book Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow - Aurélien Géron

Data Loading and Exploration:
The code loads Titanic dataset files (train.csv and test.csv), sets the PassengerId column as the index, and explores the data for missing values and basic statistics.

Data Preprocessing Pipelines:
Two preprocessing pipelines are created:

Numerical Pipeline: Handles numerical attributes by imputing missing values with median and scaling the data using StandardScaler.
Categorical Pipeline: Encodes categorical attributes using ordinal encoding, imputes missing values with the most frequent strategy, and applies one-hot encoding.
Feature Engineering:
The code defines functions (add_age_bucket and add_relatives_onboard) to create new features (AgeBucket and RelativesOnboard) based on existing attributes like age and family-related columns.

Machine Learning Models:
Random Forest Classifier: Trains a Random Forest classifier using the preprocessed data and evaluates it using cross-validation.
Support Vector Classifier (SVC): Trains an SVC classifier and evaluates its performance using cross-validation.
Evaluation and Visualization:
The code calculates and prints the mean accuracy scores of both models using cross-validation. It also visualizes the accuracy scores using a box plot for comparison.
