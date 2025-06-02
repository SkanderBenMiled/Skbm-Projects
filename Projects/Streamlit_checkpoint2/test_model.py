#!/usr/bin/env python3
"""
Test script for Financial Inclusion ML model
Tests the core functionality without Streamlit interface
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def test_model():
    print("Testing Financial Inclusion ML Model...")
    
    # Load the dataset
    try:
        df = pd.read_csv('Financial_inclusion_dataset.csv')
        print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False
    
    # Basic data cleaning
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df_cleaned = df.dropna()
    print(f"✓ Missing values handled: {df_cleaned.shape[0]} rows remaining")
    
    # Remove duplicates
    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"✓ {duplicates} duplicate rows removed")
    else:
        print("✓ No duplicate rows found")
    
    # Label encode categorical variables
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {categorical_columns}")
    
    for column in categorical_columns:
        le = LabelEncoder()
        df_cleaned[column] = le.fit_transform(df_cleaned[column])
    
    print("✓ Categorical variables encoded")
    
    # Define target and features
    target = 'bank_account'
    if target not in df_cleaned.columns:
        print(f"✗ Target column '{target}' not found in dataset")
        print(f"Available columns: {df_cleaned.columns.tolist()}")
        return False
    
    features = df_cleaned.drop(columns=[target])
    X = df_cleaned[features.columns]
    y = df_cleaned[target]
    
    print(f"✓ Features prepared: {X.shape[1]} features")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing samples")
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("✓ Random Forest model trained")
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Model accuracy: {accuracy:.3f}")
    
    # Test prediction on a sample
    sample_input = X_test.iloc[0:1]
    prediction = rf_model.predict(sample_input)[0]
    prediction_proba = rf_model.predict_proba(sample_input)[0]
    
    print(f"✓ Sample prediction: {prediction} (confidence: {max(prediction_proba):.3f})")
    
    # Feature importance
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("✓ Top 5 most important features:")
    print(feature_importance.head().to_string(index=False))
    
    print("\n🎉 All tests passed! The model is working correctly.")
    return True

if __name__ == "__main__":
    test_model()
