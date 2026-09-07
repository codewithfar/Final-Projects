# 🏦 Bank Term Deposit Subscription Prediction

## 📌 Project Overview

This project develops a machine learning classification system to predict whether a bank customer is likely to subscribe to a term deposit.

The model uses customer demographic, financial, contact and campaign-related information to generate a binary prediction:

- `0` → Customer is unlikely to subscribe
- `1` → Customer is likely to subscribe

A Streamlit web application has been developed to allow users to enter customer information and receive a prediction with the estimated subscription probability.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Perform exploratory data analysis (EDA)
- Clean and preprocess the dataset
- Handle imbalanced target classes
- Perform feature engineering
- Compare multiple machine learning algorithms
- Use ensemble/stacking techniques
- Evaluate models using multiple classification metrics
- Explain model predictions using feature importance and SHAP
- Deploy the final model through a Streamlit application

---

## 📊 Dataset

The project uses the Bank Marketing dataset.

Dataset file:

```text
data/bank-full.csv
