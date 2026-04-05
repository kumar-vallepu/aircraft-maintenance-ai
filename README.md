# Aircraft Predictive Maintenance System

An AI-based predictive maintenance system for aircraft engines that analyzes sensor data to estimate Remaining Useful Life (RUL) and predict potential failures. The system is deployed as an interactive web application for real-time analytics and model insights.

---

## Live Application

Access the deployed application:  
https://aircraft-maintenance-ai.streamlit.app

---

## Overview

This project leverages machine learning techniques to monitor aircraft engine health and predict failures before they occur. By analyzing time-series sensor data, the system enables proactive maintenance, improving operational efficiency and safety.

---

## Key Capabilities

- Failure prediction using machine learning  
- Remaining Useful Life (RUL) estimation  
- Interactive dashboard for data exploration  
- Sensor-level trend analysis  
- Model evaluation with performance metrics  
- Feature importance visualization  

---

## Technology Stack

| Component        | Technology Used            |
|-----------------|--------------------------|
| Programming     | Python                   |
| Framework       | Streamlit                |
| Machine Learning| Random Forest            |
| Data Handling   | Pandas, NumPy            |
| Visualization   | Plotly                   |

---

## Dataset

The model is trained on the NASA CMAPSS dataset, which contains:

- Engine operational settings  
- Multiple sensor measurements  
- Time-cycle data for degradation analysis  

This dataset is widely used for predictive maintenance research.

---

## Machine Learning Approach

- Model: Random Forest Classifier  
- Input Features: Selected engine sensor values  
- Target Variable: Binary failure classification  

### Outputs:
- Failure risk prediction  
- Remaining Useful Life (RUL)

---

## Model Performance

The system includes built-in evaluation metrics:

- Accuracy score  
- Confusion matrix visualization  
- Classification report  

These metrics provide insight into model reliability and prediction quality.

---

## Project Structure
  
