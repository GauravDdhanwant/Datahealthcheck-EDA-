import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
import openai
import os

# Set up page config
st.set_page_config(page_title="Advanced Data Health Check", page_icon=":bar_chart:", layout="wide")

# Load and display the logo
logo_path = "Nice Icon 3.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_column_width=True)

st.sidebar.title("Upload your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV, Excel, or Parquet file", type=["csv", "xlsx", "parquet"])

st.title("Advanced Data Health Check and Insights")

if uploaded_file is not None:
    # Data ingestion
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format")
            df = None

        if df is not None:
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Advanced EDA
            st.write("### Advanced Data Health Check")
            st.write("#### Data Shape:", df.shape)

            # Missing values
            missing_values = df.isnull().sum()
            st.write("#### Missing Values in Data:")
            st.dataframe(missing_values[missing_values > 0])

            # Data types and unique values
            st.write("#### Data Types and Unique Values:")
            st.dataframe(df.dtypes)

            # Summary statistics
            st.write("#### Summary Statistics:")
            st.dataframe(df.describe())

            # Distribution plots
            st.write("#### Distribution of Numeric Features:")
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            for col in numeric_df.columns:
                plt.figure()
                sns.histplot(df[col], kde=True)
                st.pyplot(plt)

            # Correlation matrix
            st.write("#### Correlation Matrix:")
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                st.pyplot(plt)
            else:
                st.write("No numeric columns available for correlation matrix.")

            # PCA Analysis
            st.write("### Principal Component Analysis (PCA)")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df.dropna())

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            st.write("Explained Variance Ratio by each PC:", pca.explained_variance_ratio_)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="PC1", y="PC2", data=pca_df)
            st.pyplot(plt)

            # Feature Engineering Suggestions
            st.write("### Suggested Features")
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            poly_features = poly.fit_transform(scaled_data)
            st.write("New polynomial and interaction features were generated. Consider using these for modeling.")

            # ML Use Case Suggestions using LLM
            st.write("### Suggested ML Use Cases")
            openai.api_key = os.getenv("OPENAI_API_KEY")

            # Define a prompt for the LLM
            prompt = f"""
            Given the following dataset summary: {df.describe().to_string()},
            Suggest possible machine learning use cases for this data. Consider regression, classification, clustering,
            and any other relevant tasks.
            """

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150
            )

            suggestions = response.choices[0].text.strip()
            st.write(suggestions)

            # Conversational interface
            st.write("### Ask Questions About the Data")
            user_input = st.text_input("Enter your query")
            if user_input:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"{user_input}\nData Summary: {df.describe().to_string()}",
                    max_tokens=150
                )
                st.write(response.choices[0].text.strip())
                
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a data file to proceed.")
