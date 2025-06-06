# Expresso Churn Prediction Application
# Author: Skander
# Date: June 2025

#===================================
# 1. IMPORTS AND SETUP
#===================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="📱 Expresso Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

#===================================
# 2. DATA LOADING AND EXPLORATION
#===================================
@st.cache_data
def load_and_explore_data():
    """Load and perform basic exploration of the dataset"""
    df = pd.read_csv('Expresso_churn_dataset.csv')
    
    # Basic information
    basic_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'target_distribution': df['CHURN'].value_counts().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    
    return df, basic_info

@st.cache_data
def create_profiling_report(df):
    """Create a simplified profiling report"""
    from ydata_profiling import ProfileReport
    
    # Create a sample for faster processing
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    
    profile = ProfileReport(
        df_sample, 
        title="Expresso Churn Dataset Profile",
        explorative=True,
        minimal=True
    )
    
    return profile

#===================================
# 3. DATA PREPROCESSING FUNCTIONS
#===================================
@st.cache_data
def preprocess_data(df):
    """Complete data preprocessing pipeline"""
    df_processed = df.copy()
    
    # Remove user_id as it's not needed for modeling
    if 'user_id' in df_processed.columns:
        df_processed = df_processed.drop('user_id', axis=1)
    
    # Handle missing values
    # Drop rows where target is missing
    df_processed = df_processed.dropna(subset=['CHURN'])
    
    # Fill numeric columns with median
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'CHURN']  # Exclude target
    
    for col in numeric_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Fill categorical columns with mode
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = df_processed[col].mode()
        if len(mode_value) > 0:
            df_processed[col] = df_processed[col].fillna(mode_value.iloc[0])
        else:
            df_processed[col] = df_processed[col].fillna('Unknown')
    
    # Remove duplicates
    df_processed = df_processed.drop_duplicates()
    
    # Handle outliers using IQR method
    Q1 = df_processed[numeric_cols].quantile(0.25)
    Q3 = df_processed[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Remove extreme outliers
    condition = ~((df_processed[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                  (df_processed[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_processed = df_processed[condition]
    
    return df_processed, numeric_cols, categorical_cols

@st.cache_data
def encode_features(df, _categorical_cols):
    """Encode categorical features and return encoders"""
    df_encoded = df.copy()
    encoders = {}
    
    # Create label encoders for categorical columns
    for col in _categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders

#===================================
# 4. MACHINE LEARNING FUNCTIONS
#===================================
@st.cache_data
def train_model(df_encoded):
    """Train Random Forest model"""
    # Prepare features and target
    X = df_encoded.drop('CHURN', axis=1)
    y = df_encoded['CHURN']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, accuracy, report, conf_matrix, feature_importance, X.columns

#===================================
# 5. STREAMLIT APPLICATION
#===================================

def main():
    st.title("📱 Expresso Customer Churn Prediction")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["🏠 Home", "📊 Data Exploration", "📈 Data Analysis", "🤖 Model Training", "🔮 Make Predictions"]
    )
    
    # Load data
    try:
        df, basic_info = load_and_explore_data()
        st.sidebar.success(f"✅ Data loaded successfully!")
        st.sidebar.info(f"📊 Dataset: {basic_info['shape'][0]:,} rows × {basic_info['shape'][1]} columns")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    #===================================
    # HOME PAGE
    #===================================
    if app_mode == "🏠 Home":
        st.header("Welcome to the Expresso Churn Prediction App!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Project Overview")
            st.write("""
            This application predicts customer churn for Expresso telecom company using machine learning.
            
            **Features:**
            - 📊 Comprehensive data exploration
            - 🧹 Automated data preprocessing
            - 🤖 Random Forest classification
            - 📈 Interactive visualizations
            - 🔮 Real-time predictions
            """)
            
        with col2:
            st.subheader("📊 Dataset Summary")
            st.metric("Total Records", f"{basic_info['shape'][0]:,}")
            st.metric("Features", basic_info['shape'][1] - 1)  # Excluding target
            st.metric("Target Classes", len(basic_info['target_distribution']))
            
            # Target distribution
            churn_counts = basic_info['target_distribution']
            st.write("**Churn Distribution:**")
            st.write(f"- No Churn (0): {churn_counts.get(0, 0):,}")
            st.write(f"- Churn (1): {churn_counts.get(1, 0):,}")
    
    #===================================
    # DATA EXPLORATION PAGE
    #===================================
    elif app_mode == "📊 Data Exploration":
        st.header("📊 Data Exploration")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Info", "🔍 Missing Values", "📈 Distributions", "📊 Correlations"])
        
        with tab1:
            st.subheader("📋 Dataset Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Shape:**", basic_info['shape'])
                st.write("**Duplicates:**", basic_info['duplicates'])
                
            with col2:
                st.write("**Data Types:**")
                dtypes_df = pd.DataFrame(list(basic_info['dtypes'].items()), 
                                       columns=['Column', 'Data Type'])
                st.dataframe(dtypes_df, use_container_width=True)
            
            st.subheader("🔢 First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("🔍 Missing Values Analysis")
            
            missing_df = pd.DataFrame({
                'Column': list(basic_info['missing_values'].keys()),
                'Missing Count': list(basic_info['missing_values'].values()),
                'Missing Percentage': [f"{x:.2f}%" for x in basic_info['missing_percentage'].values()]
            })
            
            # Filter only columns with missing values
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                # Visualize missing values
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_df.plot(x='Column', y='Missing Count', kind='bar', ax=ax)
                plt.xticks(rotation=45)
                plt.title('Missing Values by Column')
                st.pyplot(fig)
            else:
                st.success("✅ No missing values found in the dataset!")
        
        with tab3:
            st.subheader("📈 Feature Distributions")
            
            # Target distribution
            st.write("**Target Variable Distribution (CHURN):**")
            churn_counts = df['CHURN'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                churn_counts.plot(kind='bar', ax=ax)
                plt.title('Churn Distribution')
                plt.xlabel('Churn (0: No, 1: Yes)')
                plt.ylabel('Count')
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%')
                plt.title('Churn Distribution (Pie Chart)')
                st.pyplot(fig)
            
            # Numeric features distributions
            numeric_features = df.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col != 'CHURN']
            
            if len(numeric_features) > 0:
                st.write("**Numeric Features Distribution:**")
                selected_feature = st.selectbox("Select a numeric feature:", numeric_features)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                df[selected_feature].hist(bins=30, ax=ax)
                plt.title(f'Distribution of {selected_feature}')
                plt.xlabel(selected_feature)
                plt.ylabel('Frequency')
                st.pyplot(fig)
        
        with tab4:
            st.subheader("📊 Correlation Analysis")
            
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                correlation_matrix = numeric_df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                plt.title('Feature Correlation Matrix')
                st.pyplot(fig)
                
                # Show correlation with target
                if 'CHURN' in correlation_matrix.columns:
                    st.write("**Correlation with Target (CHURN):**")
                    churn_corr = correlation_matrix['CHURN'].sort_values(ascending=False)
                    churn_corr_df = pd.DataFrame({
                        'Feature': churn_corr.index,
                        'Correlation': churn_corr.values
                    })
                    st.dataframe(churn_corr_df, use_container_width=True)
            else:
                st.warning("⚠️ Not enough numeric features for correlation analysis.")
    
    #===================================
    # DATA ANALYSIS PAGE  
    #===================================
    elif app_mode == "📈 Data Analysis":
        st.header("📈 Advanced Data Analysis")
        
        # Generate profiling report
        with st.spinner("🔄 Generating comprehensive data profiling report..."):
            try:
                profile = create_profiling_report(df)
                
                st.subheader("📊 Pandas Profiling Report")
                st.info("📝 This report provides in-depth analysis of your dataset including distributions, correlations, and data quality insights.")
                
                # Save and display report
                profile.to_file("expresso_churn_report.html")
                
                with open("expresso_churn_report.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                st.success("✅ Profiling report generated successfully!")
                st.info("💾 Report saved as 'expresso_churn_report.html'")
                
            except Exception as e:
                st.error(f"❌ Error generating profiling report: {str(e)}")
                st.info("📊 Showing alternative analysis instead...")
                
                # Alternative analysis
                st.subheader("📈 Feature Analysis by Churn")
                
                # Categorical features analysis
                categorical_features = df.select_dtypes(include=['object']).columns
                
                if len(categorical_features) > 0:
                    selected_cat_feature = st.selectbox("Select categorical feature:", categorical_features)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    churn_by_feature = pd.crosstab(df[selected_cat_feature], df['CHURN'], normalize='index')
                    churn_by_feature.plot(kind='bar', ax=ax)
                    plt.title(f'Churn Rate by {selected_cat_feature}')
                    plt.xlabel(selected_cat_feature)
                    plt.ylabel('Proportion')
                    plt.legend(['No Churn', 'Churn'])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
    
    #===================================
    # MODEL TRAINING PAGE
    #===================================
    elif app_mode == "🤖 Model Training":
        st.header("🤖 Machine Learning Model Training")
        
        with st.spinner("🔄 Processing data and training model..."):
            try:
                # Preprocess data
                df_processed, numeric_cols, categorical_cols = preprocess_data(df)
                st.success(f"✅ Data preprocessing completed! Shape: {df_processed.shape}")
                
                # Encode features
                df_encoded, encoders = encode_features(df_processed, categorical_cols)
                st.success("✅ Feature encoding completed!")
                
                # Train model
                model, accuracy, report, conf_matrix, feature_importance, feature_names = train_model(df_encoded)
                st.success(f"✅ Model training completed! Accuracy: {accuracy:.4f}")
                
                # Save model and encoders
                joblib.dump(model, 'churn_model.pkl')
                joblib.dump(encoders, 'encoders.pkl')
                joblib.dump(list(feature_names), 'feature_names.pkl')
                st.success("✅ Model and encoders saved successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Model Performance")
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision (Class 1)", f"{report['1']['precision']:.4f}")
                    st.metric("Recall (Class 1)", f"{report['1']['recall']:.4f}")
                    st.metric("F1-Score (Class 1)", f"{report['1']['f1-score']:.4f}")
                
                with col2:
                    st.subheader("🎯 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)
                
                st.subheader("⭐ Feature Importance")
                fig, ax = plt.subplots(figsize=(12, 8))
                top_features = feature_importance.head(15)
                sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
                plt.title('Top 15 Most Important Features')
                plt.xlabel('Importance')
                st.pyplot(fig)
                
                st.dataframe(feature_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
    
    #===================================
    # PREDICTION PAGE
    #===================================
    elif app_mode == "🔮 Make Predictions":
        st.header("🔮 Customer Churn Prediction")
        
        try:
            # Load model and encoders
            model = joblib.load('churn_model.pkl')
            encoders = joblib.load('encoders.pkl')
            feature_names = joblib.load('feature_names.pkl')
            st.success("✅ Model loaded successfully!")
            
            st.subheader("📝 Enter Customer Information")
            
            # Create input form
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    region = st.selectbox("Region", ["DAKAR", "FATICK", "THIES", "Unknown"])
                    tenure = st.selectbox("Tenure", ["K > 24 month", "I 18-21 month", "Unknown"])
                    montant = st.number_input("Montant", min_value=0.0, value=0.0)
                    frequence_rech = st.number_input("Recharge Frequency", min_value=0.0, value=0.0)
                    revenue = st.number_input("Revenue", min_value=0.0, value=0.0)
                    arpu_segment = st.number_input("ARPU Segment", min_value=0.0, value=0.0)
                
                with col2:
                    frequence = st.number_input("Frequency", min_value=0.0, value=0.0)
                    data_volume = st.number_input("Data Volume", min_value=0.0, value=0.0)
                    on_net = st.number_input("On Net Usage", min_value=0.0, value=0.0)
                    orange = st.number_input("Orange Usage", min_value=0.0, value=0.0)
                    tigo = st.number_input("Tigo Usage", min_value=0.0, value=0.0)
                    zone1 = st.number_input("Zone1", min_value=0.0, value=0.0)
                
                with col3:
                    zone2 = st.number_input("Zone2", min_value=0.0, value=0.0)
                    mrg = st.selectbox("MRG", ["YES", "NO"])
                    regularity = st.number_input("Regularity", min_value=0.0, value=0.0)
                    top_pack = st.selectbox("Top Pack", ["On net 200F=Unlimited _call24H", "Data:1000F=5GB,7d", "Unknown"])
                    freq_top_pack = st.number_input("Freq Top Pack", min_value=0.0, value=0.0)
                
                submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
                
                if submitted:
                    try:
                        # Create input dataframe
                        input_data = pd.DataFrame([[
                            region, tenure, montant, frequence_rech, revenue, arpu_segment,
                            frequence, data_volume, on_net, orange, tigo, zone1, zone2,
                            mrg, regularity, top_pack, freq_top_pack
                        ]], columns=feature_names)
                        
                        # Encode categorical features
                        for col in input_data.columns:
                            if col in encoders:
                                le = encoders[col]
                                try:
                                    input_data[col] = le.transform(input_data[col].astype(str))
                                except ValueError:
                                    # Handle unseen categories
                                    input_data[col] = 0  # Default encoding for unseen categories
                        
                        # Make prediction
                        prediction = model.predict(input_data)[0]
                        prediction_proba = model.predict_proba(input_data)[0]
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if prediction == 1:
                                st.error("⚠️ **HIGH RISK**: Customer likely to churn")
                                st.metric("Churn Probability", f"{prediction_proba[1]:.2%}")
                            else:
                                st.success("✅ **LOW RISK**: Customer likely to stay")
                                st.metric("Retention Probability", f"{prediction_proba[0]:.2%}")
                        
                        with col2:
                            # Probability chart
                            fig, ax = plt.subplots(figsize=(8, 6))
                            categories = ['No Churn', 'Churn']
                            probabilities = prediction_proba
                            colors = ['green', 'red']
                            
                            bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
                            ax.set_ylabel('Probability')
                            ax.set_title('Churn Prediction Probabilities')
                            ax.set_ylim(0, 1)
                            
                            # Add percentage labels on bars
                            for bar, prob in zip(bars, probabilities):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')
                            
                            st.pyplot(fig)
                        
                        # Recommendations
                        st.subheader("💡 Recommendations")
                        if prediction == 1:
                            st.write("""
                            **Retention Strategies:**
                            - 🎁 Offer personalized promotions or discounts
                            - 📞 Reach out with customer service call
                            - 📱 Provide better data packages or services
                            - 🎯 Target with loyalty programs
                            - 📊 Monitor usage patterns more closely
                            """)
                        else:
                            st.write("""
                            **Customer Success:**
                            - ✅ Customer shows good retention signals
                            - 🌟 Consider upselling opportunities
                            - 📈 Monitor for continued satisfaction
                            - 💝 Maintain current service quality
                            """)
                            
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
            
            # Feature importance for this prediction
            st.subheader("📊 Model Insights")
            st.info("💡 The model uses Random Forest algorithm and considers the following top features for prediction:")
            
            # Load and display feature importance
            try:
                df_processed, _, _ = preprocess_data(df)
                df_encoded, _ = encode_features(df_processed, df.select_dtypes(include=['object']).columns)
                _, _, _, _, feature_importance, _ = train_model(df_encoded)
                
                top_features = feature_importance.head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
                plt.title('Top 10 Most Important Features for Churn Prediction')
                plt.xlabel('Feature Importance')
                st.pyplot(fig)
                
            except Exception as e:
                st.warning("⚠️ Could not load feature importance visualization")
        
        except FileNotFoundError:
            st.warning("⚠️ Model not found! Please train the model first in the 'Model Training' section.")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")

if __name__ == "__main__":
    main()