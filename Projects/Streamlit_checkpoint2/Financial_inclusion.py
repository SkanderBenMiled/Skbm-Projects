import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Load the dataset
df = pd.read_csv('Financial_inclusion_dataset.csv')
# Ydata profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title='Financial Inclusion Dataset Profiling Report', explorative=True)
# Display the profiling report in Streamlit
st.title('Financial Inclusion Dataset Profiling Report')
st.write("This report provides an overview of the financial inclusion dataset, including data types, missing values, and basic statistics.")
st.write("The dataset is sourced from the World Bank and includes data on financial inclusion indicators such as account ownership, mobile money usage, and more.")
st.write("The dataset contains the following columns:")
st.write(df.columns.tolist())
st.write("The dataset contains the following number of rows and columns:")
st.write(f'The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.')
st.write("The dataset contains the following data types:")
st.write(df.dtypes)
# Display the first few rows of the dataset
st.title('Financial Inclusion Dashboard')
st.write("This dashboard provides insights into financial inclusion in various countries based on the dataset.")
st.write("The dataset contains information on financial inclusion indicators such as account ownership, mobile money usage, and more.")
st.write("The dataset is sourced from the World Bank and includes data from various countries over several years.")
st.write("The dataset contains the following columns:")
st.write(df.columns.tolist())
# Display the first few rows of the dataset
st.subheader('Dataset Overview')
try:
    st.dataframe(df.head())
except Exception as e:
    st.warning(f"Cannot display DataFrame: {e}")
    st.write("Dataset shape:", df.shape)
    st.write("Column names:", df.columns.tolist())
# Display the shape of the dataset
st.subheader('Dataset Shape')
st.write(f'The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.')
# Display the data types of the columns
st.subheader('Data Types')
st.write(df.dtypes)
# Display the summary statistics of the dataset
st.subheader('Summary Statistics')
st.write(df.describe())
# Handling missing values
st.subheader('Missing Values')
st.write("The dataset contains missing values. We will handle them by dropping rows with missing values.")
df_cleaned = df.dropna()
st.write(f'The cleaned dataset contains {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns after dropping rows with missing values.')
# Display the cleaned dataset
st.subheader('Cleaned Dataset Overview')
try:
    st.dataframe(df_cleaned.head())
except Exception as e:
    st.warning(f"Cannot display DataFrame: {e}")
    st.write("Cleaned dataset shape:", df_cleaned.shape)
# handling duplicates
st.subheader('Handling Duplicates')
st.write("The dataset may contain duplicate rows. We will check for duplicates and remove them if necessary.")
duplicates = df_cleaned.duplicated().sum()
st.write(f'The dataset contains {duplicates} duplicate rows.')
if duplicates > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    st.write(f'Duplicate rows have been removed. The cleaned dataset now contains {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns.')
else:
    st.write("No duplicate rows found in the dataset.")
# Display the cleaned dataset after handling duplicates
st.subheader('Cleaned Dataset After Handling Duplicates')
try:
    st.dataframe(df_cleaned.head())
except Exception as e:
    st.warning(f"Cannot display DataFrame: {e}")
    st.write("Dataset shape after duplicate removal:", df_cleaned.shape)
# Handling outliers
st.subheader('Handling Outliers')
st.write("We will check for outliers in the dataset using box plots and remove them if necessary.")
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
# Display box plots for numerical columns to identify outliers
numerical_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
for column in numerical_columns:
    st.subheader(f'Box Plot for {column}')
    fig, ax = plt.subplots()
    sns.boxplot(x=df_cleaned[column], ax=ax)
    st.pyplot(fig)
    # Remove outliers for the current column
    df_cleaned = remove_outliers(df_cleaned, column)
    st.write(f'Outliers removed for {column}. The cleaned dataset now contains {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns.')
# Display the cleaned dataset after handling outliers
st.subheader('Cleaned Dataset After Handling Outliers')
try:
    st.dataframe(df_cleaned.head())
except Exception as e:
    st.warning(f"Cannot display DataFrame: {e}")
    st.write("Dataset shape after outlier removal:", df_cleaned.shape)
# Label encoding for categorical variables
st.subheader('Label Encoding for Categorical Variables')
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
# Store original categorical columns before encoding
original_categorical_columns = categorical_columns.copy()
for column in categorical_columns:
    st.write(f'Encoding {column}...')
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    st.write(f'{column} encoded successfully.')
# Display the cleaned dataset after label encoding
st.subheader('Cleaned Dataset After Label Encoding')
# Display basic info about the encoded dataset
st.write("Dataset shape:", df_cleaned.shape)
st.write("Column data types after encoding:")
st.write(df_cleaned.dtypes)

# Try to display first few rows with error handling
try:
    st.write("First 5 rows:")
    st.dataframe(df_cleaned.head())
except Exception as e:
    st.warning(f"Cannot display DataFrame due to serialization issue: {e}")
    st.write("Dataset statistics instead:")
    st.write(df_cleaned.describe())
# Display the cleaned dataset shape after all preprocessing steps
st.subheader('Final Cleaned Dataset Shape')
st.write(f'The final cleaned dataset contains {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns after all preprocessing steps.')
# Display the final cleaned dataset
st.subheader('Final Cleaned Dataset Overview')
try:
    st.dataframe(df_cleaned.head())
except Exception as e:
    st.warning(f"Cannot display DataFrame: {e}")
    st.write("Final dataset shape:", df_cleaned.shape)
    st.write("Columns:", df_cleaned.columns.tolist())
# Machine Learning Model
st.subheader('Machine Learning Model')
st.write("We will build a simple machine learning model to predict financial inclusion based on the cleaned dataset.")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Define the target variable and features
st.write('df_cleaned columns:', df_cleaned.columns.tolist())
# Define the target variable and features
# Make sure the target column name matches exactly
# For example, if the column is 'account_ownership', this will work
# If not, update the target variable below to match the actual column name
# Example: target = 'Account_Ownership' if that's the real name

target = 'bank_account'  # Correct target variable name
if target not in df_cleaned.columns:
    st.error(f"Target column '{target}' not found in df_cleaned columns. Please check the column name.")
    st.stop()
features = df_cleaned.drop(columns=[target])
X = df_cleaned[features.columns]
y = df_cleaned[target]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = rf_model.predict(X_test)
st.subheader('Model Evaluation')
st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))
st.write('Confusion Matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
# Display feature importance
st.subheader('Feature Importance')
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
st.write(feature_importance_df)
# --- Streamlit Prediction Form ---
st.subheader('Make a Prediction')

# Get feature names (excluding target)
input_features = features.columns.tolist()

# Create input fields dynamically for each feature
user_input = {}
for col in input_features:
    if col in original_categorical_columns:
        # For originally categorical columns, show unique encoded values
        options = sorted(df_cleaned[col].unique().tolist())
        user_input[col] = st.selectbox(f"Select {col} (encoded)", options)
    else:
        # For numerical, use number_input
        min_val = float(df_cleaned[col].min())
        max_val = float(df_cleaned[col].max())
        mean_val = float(df_cleaned[col].mean())
        user_input[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=mean_val)

# Predict button
if st.button('Predict'):
    # Prepare input for prediction
    input_df = pd.DataFrame([user_input])
    # Ensure columns are in the same order as training
    input_df = input_df[input_features]
    # Predict
    prediction = rf_model.predict(input_df)[0]
    prediction_proba = rf_model.predict_proba(input_df)[0]
    
    # Convert prediction back to meaningful text
    if prediction == 1:
        result = "Yes (Has Bank Account)"
    else:
        result = "No (No Bank Account)"
    
    st.success(f'Predicted bank account ownership: {result}')
    st.info(f'Prediction confidence: {max(prediction_proba):.2%}')