import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load logo and display it
st.set_page_config(page_title="Advanced Data Health Check", page_icon=":bar_chart:", layout="wide")

# Load and display the logo
logo = Image.open("assets/logo.png")  # Make sure to place your logo in the 'assets' folder
st.sidebar.image(logo, use_column_width=True)

st.sidebar.title("Upload your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV, Excel, or Parquet file", type=["csv", "xlsx", "parquet"])

st.title("Advanced Data Health Check and Insights")

if uploaded_file is not None:
    # Data ingestion
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
        corr_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)

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
else:
    st.info("Please upload a data file to proceed.")
