# Sampling and Feature Engineering with Two Datasets

This project demonstrates **sampling** and **feature engineering** techniques using two different datasets:
1. **Pima Indians Diabetes Dataset** for **stratified sampling**.
2. **California Housing Dataset** for **feature engineering**.

The project includes:
- Handling missing data.
- Scaling numeric features.
- Performing stratified and random sampling.
- Visualizing results using histograms, heatmaps, and scatter plots.
- Providing an interactive Tkinter GUI to display the visualizations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Tasks Performed](#tasks-performed)
6. [Dependencies](#dependencies)
7. [License](#license)

---

## Project Overview

The project is divided into two parts:
1. **Stratified Sampling with Pima Indians Diabetes Dataset**:
   - Perform stratified sampling using Scikit-learn's `StratifiedShuffleSplit`.
   - Analyze class distributions in training and testing sets.
2. **Feature Engineering with California Housing Dataset**:
   - Handle missing data by identifying and imputing missing values.
   - Scale numeric features using Scikit-learn's `StandardScaler`.
   - Visualize the results using histograms, heatmaps, and scatter plots.

---

## Features

### Stratified Sampling (Pima Indians Diabetes Dataset)
- **Stratified Sampling**:
  - Use Scikit-learn's `StratifiedShuffleSplit` to maintain class distribution in training and testing sets.
- **Random Sampling**:
  - Use Scikit-learn's `train_test_split` for random sampling.
- **Class Distribution Analysis**:
  - Verify class distributions in training and testing sets.

### Feature Engineering (California Housing Dataset)
- **Handling Missing Data**:
  - Identify missing values in the dataset.
  - Impute missing values using the median for numeric columns.
- **Feature Scaling**:
  - Scale numeric features using Scikit-learn's `StandardScaler` to standardize them (mean = 0, standard deviation = 1).
- **Visualizations**:
  - Generate histograms to show feature distributions.
  - Create a correlation heatmap to analyze feature relationships.
  - Plot scatter plots to visualize relationships between features and the target variable.

### Interactive GUI
- Use Tkinter to create a GUI with buttons for displaying visualizations.

---

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/sampling-feature-engineering.git
   cd sampling-feature-engineering

2. Install the required packages:
    pip install -r requirements.txt

3. Run main file in this case:
    python feature_engineering.py
    python stratified_sampling.py



### **Key Features of the `README.md`**
1. **Combines Both Datasets**:
   - Clearly separates the tasks for the **Pima Indians Diabetes Dataset** (sampling) and the **California Housing Dataset** (feature engineering).
2. **Detailed Task Breakdown**:
   - Provides a step-by-step breakdown of tasks for both datasets.
3. **Interactive GUI**:
   - Highlights the use of Tkinter for visualizing results.
4. **Easy Setup**:
   - Includes clear installation and usage instructions.
