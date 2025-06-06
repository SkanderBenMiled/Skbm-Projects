#!/usr/bin/env python3
"""
Demo script to show encoding mappings for the Financial Inclusion dataset
This helps users understand how categorical values are encoded
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def show_encoding_mappings():
    print("🔍 Financial Inclusion Dataset - Encoding Mappings Demo")
    print("=" * 60)
    
    # Load the dataset with proper path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'Financial_inclusion_dataset.csv')
    df = pd.read_csv(csv_path)
    print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Clean the data
    df_cleaned = df.dropna()
    print(f"✨ After cleaning: {df_cleaned.shape[0]} rows")
    
    # Get categorical columns
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    print(f"\n📋 Found {len(categorical_columns)} categorical columns:")
    
    for i, col in enumerate(categorical_columns, 1):
        print(f"  {i}. {col}")
    
    print("\n" + "=" * 60)
    print("🔢 ENCODING MAPPINGS")
    print("=" * 60)
    
    # Show encoding mappings for each categorical column
    for column in categorical_columns:
        print(f"\n📌 {column.upper()}:")
        print("-" * 40)
        
        # Get unique values
        unique_values = sorted(df_cleaned[column].unique())
        
        # Create and fit label encoder
        le = LabelEncoder()
        encoded_values = le.fit_transform(unique_values)
        
        # Create mapping
        mapping = dict(zip(unique_values, encoded_values))
        
        # Display mapping in a nice format
        for original, encoded in mapping.items():
            print(f"  '{original}' → {encoded}")
        
        print(f"  Total unique values: {len(unique_values)}")
    
    print("\n" + "=" * 60)
    print("💡 HOW TO USE THIS INFORMATION:")
    print("=" * 60)
    print("""
When using the Streamlit prediction interface:

1. The app shows you the ORIGINAL values (like 'Kenya', 'Male', 'Rural')
2. You select from these original, human-readable options
3. The app automatically converts them to encoded numbers for the model
4. You get predictions based on the encoded values
5. Results are shown in plain English

This makes the app much more user-friendly than asking users to 
remember that 'Kenya' = 1, 'Male' = 0, etc.
""")
    
    print("🎯 Example prediction workflow:")
    print("  User selects: Country = 'Kenya', Gender = 'Male', Location = 'Urban'")
    print("  App encodes:  Country = 1, Gender = 0, Location = 1")
    print("  Model predicts: Bank Account = 1 (Yes)")
    print("  User sees: '✅ YES - Likely to have a bank account (85% confidence)'")

if __name__ == "__main__":
    show_encoding_mappings()
