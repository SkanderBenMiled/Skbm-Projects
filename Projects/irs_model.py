import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# Load the Iris dataset

iris = load_iris()
x = iris.data
y = iris.target
# Convert to DataFrame
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
# Predict on the test set
y_pred = model.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
# Streamlit app
st.title(f"Iris Dataset Dashboard - Model Accuracy: {accuracy:.2f}")
st.write("Enter the flower measurements to predict the species:")
sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.slider("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
if st.button("Predict Species"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    st.write(f"The predicted species is: **{iris.target_names[prediction[0]]}**")
    st.subheader("Prediction Details")
    st.write(f"the predicted iris species is **{iris.target_names[prediction][0]}**")
    st.subheader("Prediction probabilities")
    prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.dataframe(prob_df)



