# Financial Inclusion Data Science Application

## Overview
This Streamlit application provides comprehensive analysis of financial inclusion data, including data exploration, preprocessing, machine learning model training, and interactive predictions.

## Features
- **Data Exploration**: Overview of the dataset with basic statistics and profiling
- **Data Preprocessing**: 
  - Missing value handling
  - Duplicate removal
  - Outlier detection and removal using IQR method
  - Label encoding for categorical variables
- **Encoding Reference**: Interactive mappings showing original categorical values and their encoded numbers
- **Machine Learning**: Random Forest classifier for predicting bank account ownership
- **Enhanced Prediction Interface**: 
  - User-friendly form with original categorical values (not just encoded numbers)
  - Real-time input summary
  - Detailed prediction results with confidence levels
- **Visualizations**: Box plots, confusion matrix, and feature importance charts
- **Feature Explanations**: Detailed descriptions of what each feature represents

## Dataset
The application uses a financial inclusion dataset containing information about:
- Bank account ownership (target variable)
- Demographics (age, gender, education, location)
- Household characteristics
- Access to technology (cellphone access)

## How to Run
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```
   streamlit run Financial_inclusion.py
   ```
3. Open your browser to `http://localhost:8501`

## Model Performance
The Random Forest classifier provides:
- Accuracy metrics
- Classification report
- Confusion matrix visualization
- Feature importance analysis

## Deployment
This application can be deployed on Streamlit Share by:
1. Pushing the code to a GitHub repository
2. Connecting the repository to Streamlit Share
3. The app will be automatically deployed and accessible via a public URL

## Files
- `Financial_inclusion.py` - Main Streamlit application
- `Financial_inclusion_dataset.csv` - Dataset file
- `requirements.txt` - Python dependencies
- `README.md` - This documentation file
