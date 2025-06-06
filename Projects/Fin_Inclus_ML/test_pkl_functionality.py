"""
Test script to demonstrate .pkl file functionality
This shows how the model saving and loading works
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

print("🔍 Testing .pkl file functionality...")
print("=" * 50)

# Load the dataset
try:
    df = pd.read_csv('Financial_inclusion_dataset.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Quick preprocessing
print("\n🧹 Preprocessing data...")
df_cleaned = df.dropna()
print(f"📊 After dropping NaN: {df_cleaned.shape}")

# Encode categorical variables
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
encoding_mappings = {}

for column in categorical_columns:
    le = LabelEncoder()
    original_values = df_cleaned[column].unique()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    
    # Store encoder and mappings
    label_encoders[column] = le
    encoding_mappings[column] = {
        orig: enc for orig, enc in zip(original_values, le.transform(original_values))
    }

print(f"✅ Encoded {len(categorical_columns)} categorical columns")

# Prepare features and target
target = 'bank_account'
if target not in df_cleaned.columns:
    print(f"❌ Target column '{target}' not found!")
    print(f"Available columns: {df_cleaned.columns.tolist()}")
    exit()

features = df_cleaned.drop(columns=[target])
X = features
y = df_cleaned[target]

# Train model
print(f"\n🤖 Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Smaller for speed
rf_model.fit(X_train, y_train)

accuracy = rf_model.score(X_test, y_test)
print(f"✅ Model trained! Accuracy: {accuracy:.3f}")

# Save everything to .pkl files
print(f"\n💾 Saving model and preprocessing components...")
try:
    joblib.dump(rf_model, 'financial_inclusion_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(encoding_mappings, 'encoding_mappings.pkl')
    joblib.dump(list(features.columns), 'feature_names.pkl')
    print("✅ All .pkl files saved successfully!")
except Exception as e:
    print(f"❌ Error saving files: {e}")

# Check file sizes
print(f"\n📁 .pkl file information:")
pkl_files = [
    'financial_inclusion_model.pkl',
    'label_encoders.pkl', 
    'encoding_mappings.pkl',
    'feature_names.pkl'
]

for file in pkl_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   📄 {file}: {size:,} bytes ({size/1024:.1f} KB)")
    else:
        print(f"   ❌ {file}: Not found")

# Test loading the saved files
print(f"\n🔄 Testing model loading...")
try:
    loaded_model = joblib.load('financial_inclusion_model.pkl')
    loaded_encoders = joblib.load('label_encoders.pkl')
    loaded_mappings = joblib.load('encoding_mappings.pkl')
    loaded_features = joblib.load('feature_names.pkl')
    print("✅ All files loaded successfully!")
    
    # Test prediction with loaded model
    test_accuracy = loaded_model.score(X_test, y_test)
    print(f"✅ Loaded model accuracy: {test_accuracy:.3f}")
    
    # Show what's inside the encoders
    print(f"\n🔍 What's inside the .pkl files:")
    print(f"   🤖 Model: {type(loaded_model).__name__} with {loaded_model.n_estimators} trees")
    print(f"   🏷️  Encoders: {len(loaded_encoders)} LabelEncoder objects")
    print(f"   📋 Mappings: {len(loaded_mappings)} categorical mappings")
    print(f"   📊 Features: {len(loaded_features)} feature names")
    
    # Show a sample encoding mapping
    if loaded_mappings:
        sample_column = list(loaded_mappings.keys())[0]
        print(f"\n📝 Sample encoding mapping for '{sample_column}':")
        for orig, enc in list(loaded_mappings[sample_column].items())[:5]:
            print(f"      '{orig}' → {enc}")
        if len(loaded_mappings[sample_column]) > 5:
            print(f"      ... and {len(loaded_mappings[sample_column])-5} more")
            
except Exception as e:
    print(f"❌ Error loading files: {e}")

print(f"\n🎉 .pkl functionality test complete!")
print("=" * 50)
