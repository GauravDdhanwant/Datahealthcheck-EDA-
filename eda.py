import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set up page config
st.set_page_config(page_title="Advanced Data Health Check", page_icon=":bar_chart:", layout="wide")

# Try to load and display the logo
logo_path = "Nice Icon 3 (1).png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_column_width=True)
else:
    st.sidebar.write("Logo not found. Please ensure 'logo.png' is in the 'assets' directory.")

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

            # Data health check
            st.write("### Data Health Check")
            st.write("#### Data Shape:", df.shape)

            # Missing values
            missing_values = df.isnull().sum()
            st.write("#### Missing Values in Data:")
            st.dataframe(missing_values[missing_values > 0])

            # Data types and unique values
            st.write("#### Data Types and Unique Values:")
            st.dataframe(df.dtypes)

            # Correlation matrix
            st.write("#### Correlation Matrix:")
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                st.pyplot(plt)
            else:
                st.write("No numeric columns available for correlation matrix.")

            # Insights and Use Cases
            st.write("### Insights and Suggested Use Cases")
            if 'target' in df.columns:
                target_type = df['target'].dtype
                if pd.api.types.is_numeric_dtype(target_type):
                    st.write("This dataset is suitable for Regression tasks.")
                elif pd.api.types.is_categorical_dtype(target_type):
                    st.write("This dataset is suitable for Classification tasks.")
            else:
                st.write("Unsupervised learning techniques like Clustering or Anomaly Detection could be applied.")

            # Conversational interface
            st.write("### Ask Questions About the Data")
            user_input = st.text_input("Enter your query")
            if user_input:
                if "correlation" in user_input.lower():
                    st.write("You asked for correlation analysis. Displaying correlation matrix again:")
                    st.pyplot(plt)
                elif "missing values" in user_input.lower():
                    st.write("Displaying columns with missing values:")
                    st.dataframe(missing_values[missing_values > 0])
                else:
                    st.write("Query not recognized. Please try asking about correlation, missing values, etc.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a data file to proceed.")
