import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport  # or `ydata_profiling`
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"data\Expresso_churn_dataset.csv")

# Basic exploration
print(df.head())
print(df.info())
print(df.describe(include='all'))
print(df['CHURN'].value_counts())  # Target variable

print(df.columns)
print(df.shape)
#pandas profiling
#profile = ProfileReport(df, title="Expresso Churn Profiling Report", explorative=True)
#profile.to_file("expresso_churn_report.html")

# Check missing values
print(df.isnull().sum())


# Check for missing values
print(df.isnull().sum() / len(df) * 100)

# Example handling
df.dropna(subset=['CHURN'], inplace=True)  # Drop if target missing

# Fill numeric NaNs with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical NaNs with mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


#remove dupplicates
df.drop_duplicates(inplace=True)

#Outliers handling
# Only apply IQR filtering on numeric columns
# Use IQR to remove extreme outliers in key numeric columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]



#Encode Categorical Variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


X = df.drop(['user_id', 'CHURN'], axis=1)
y = df['CHURN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'churn_model.pkl')


# Load model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

# Input fields
REGION = st.selectbox("Region", ["DAKAR", "FATICK", "THIES", "NaN"])
TENURE = st.selectbox("Tenure", ["K > 24 month", "I 18-21 month"])
MONTANT = st.number_input("Montant", min_value=0)
FREQUENCE_RECH = st.number_input("Recharge Frequency", min_value=0)
REVENUE = st.number_input("Revenue", min_value=0)
ARPU_SEGMENT = st.number_input("ARPU Segment", min_value=0)
FREQUENCE = st.number_input("Frequency", min_value=0)
DATA_VOLUME = st.number_input("Data Volume", min_value=0)
ON_NET = st.number_input("On Net Usage", min_value=0)
ORANGE = st.number_input("Orange Usage", min_value=0)
TIGO = st.number_input("Tigo Usage", min_value=0)
ZONE1 = st.number_input("Zone1", min_value=0)
ZONE2 = st.number_input("Zone2", min_value=0)
MRG = st.selectbox("MRG", ["YES", "NO"])
REGULARITY = st.number_input("Regularity", min_value=0)
TOP_PACK = st.selectbox("Top Pack", ["On net 200F=Unlimited _call24H", "Data:1000F=5GB,7d"])
FREQ_TOP_PACK = st.number_input("Freq Top Pack", min_value=0)

if st.button("Predict Churn"):
    input_data = pd.DataFrame([[REGION, TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT,
                                 FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO, ZONE1, ZONE2,
                                 MRG, REGULARITY, TOP_PACK, FREQ_TOP_PACK]],
                               columns=['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
                                        'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET',
                                        'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'MRG', 'REGULARITY',
                                        'TOP_PACK', 'FREQ_TOP_PACK'])

    # Encode
    for col in input_data.columns:
        input_data[col] = le.fit(df[col]).transform(input_data[col])

    prediction = model.predict(input_data)
    st.success(f"The predicted churn status is: {'Churn' if prediction[0]==1 else 'No Churn'}")


