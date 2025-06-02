import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Load the dataset with proper path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Financial_inclusion_dataset.csv')
df = pd.read_csv(csv_path)
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

# Store the original data and label encoders for later use
original_data = df.copy()
label_encoders = {}
encoding_mappings = {}

for column in categorical_columns:
    st.write(f'Encoding {column}...')
    le = LabelEncoder()
    
    # Store original unique values
    original_values = df_cleaned[column].unique()
    
    # Fit and transform
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    
    # Store the encoder and mapping
    label_encoders[column] = le
    encoding_mappings[column] = {
        original_val: encoded_val 
        for original_val, encoded_val in zip(original_values, le.transform(original_values))
    }
    
    st.write(f'{column} encoded successfully.')
    
# Display encoding mappings
st.subheader('🔍 Encoding Mappings Reference')
st.write("Here are the mappings between original categorical values and their encoded numbers:")

for column in original_categorical_columns:
    with st.expander(f"📋 {column} Mappings"):
        mapping_df = pd.DataFrame([
            {'Original Value': orig, 'Encoded Value': enc}
            for orig, enc in encoding_mappings[column].items()
        ]).sort_values('Encoded Value')
        st.dataframe(mapping_df, use_container_width=True)
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

# Add explanation section
st.subheader('📚 Understanding the Features')
st.write("""
This model uses the following features to predict bank account ownership:

**Demographic Features:**
- **Age of Respondent**: Age of the person being surveyed
- **Gender of Respondent**: Male or Female
- **Education Level**: Highest level of education completed
- **Marital Status**: Current marital status

**Geographic Features:**
- **Country**: Country where the person lives
- **Location Type**: Rural or Urban location

**Economic Features:**
- **Job Type**: Type of employment or economic activity
- **Household Size**: Number of people in the household

**Technology Access:**
- **Cellphone Access**: Whether the person has access to a cellphone

**Social Features:**
- **Relationship with Head**: Relationship to the head of household

**Temporal:**
- **Year**: Year of the survey

Each categorical feature is encoded with numbers where each unique value gets a specific number (see the encoding mappings above).
""")
# --- Enhanced Streamlit Prediction Form ---
st.subheader('🎯 Make a Prediction')
st.write("Use this form to predict whether someone is likely to have a bank account based on their characteristics.")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Input Features")
    
    # Get feature names (excluding target)
    input_features = features.columns.tolist()
    
    # Create input fields dynamically for each feature
    user_input = {}
    user_input_display = {}  # Store display values for showing to user
    
    for col in input_features:
        if col in original_categorical_columns:
            # For originally categorical columns, show original values
            st.write(f"**{col.replace('_', ' ').title()}:**")
            
            # Get unique original values
            original_values = list(encoding_mappings[col].keys())
            
            # Create selectbox with original values
            selected_original = st.selectbox(
                f"Choose {col.replace('_', ' ')}",
                options=original_values,
                key=f"input_{col}"
            )
            
            # Store both original and encoded values
            user_input[col] = encoding_mappings[col][selected_original]
            user_input_display[col] = f"{selected_original} (encoded as {encoding_mappings[col][selected_original]})"
            
        else:
            # For numerical, use number_input
            min_val = float(df_cleaned[col].min())
            max_val = float(df_cleaned[col].max())
            mean_val = float(df_cleaned[col].mean())
            
            user_input[col] = st.number_input(
                f"**{col.replace('_', ' ').title()}**", 
                min_value=min_val, 
                max_value=max_val, 
                value=mean_val,
                help=f"Range: {min_val:.1f} to {max_val:.1f}"
            )
            user_input_display[col] = str(user_input[col])

with col2:
    st.write("### 📊 Input Summary")
    
    # Show what the user has selected
    for feature, display_value in user_input_display.items():
        st.write(f"**{feature.replace('_', ' ').title()}:** {display_value}")

# Prediction section
st.write("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button('🔮 Predict Bank Account Ownership', use_container_width=True)

if predict_button:
    # Prepare input for prediction
    input_df = pd.DataFrame([user_input])
    # Ensure columns are in the same order as training
    input_df = input_df[input_features]
    
    # Predict
    prediction = rf_model.predict(input_df)[0]
    prediction_proba = rf_model.predict_proba(input_df)[0]
    
    # Convert prediction back to meaningful text
    if prediction == 1:
        result = "✅ **YES** - Likely to have a bank account"
        result_color = "success"
    else:
        result = "❌ **NO** - Unlikely to have a bank account"
        result_color = "error"
    
    # Display results with confidence
    confidence = max(prediction_proba)
    
    if result_color == "success":
        st.success(result)
    else:
        st.error(result)
    
    # Show confidence with appropriate color coding
    if confidence >= 0.8:
        st.info(f'🎯 **Prediction Confidence:** {confidence:.1%} (High confidence)')
    elif confidence >= 0.6:
        st.warning(f'🎯 **Prediction Confidence:** {confidence:.1%} (Medium confidence)')
    else:
        st.warning(f'🎯 **Prediction Confidence:** {confidence:.1%} (Low confidence - prediction uncertain)')
    
    # Show probability breakdown
    prob_no = prediction_proba[0] * 100
    prob_yes = prediction_proba[1] * 100
    
    st.write("### 📈 Probability Breakdown:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("No Bank Account", f"{prob_no:.1f}%")
    with col2:
        st.metric("Has Bank Account", f"{prob_yes:.1f}%")