import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras
import tensorflow as tf

# === Load pre-trained models and preprocessors ===
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')

# === Define default values for non-essential features for the full model ===
default_values = {
  'Age': 30,  # Will be overwritten by user input
  'Span ft': 300,  # Will be overwritten by user input
  'Deck Width ft': 20,  # Will be overwritten by user input
  'Condition Rating': 4,
  'Num Lanes': 6,  # Will be overwritten by user input
  'Material': 'Steel',  # Will be overwritten by user input
}

st.title("Lab 11 Bridge Data")

# Sidebar: let the user choose which model to use
model_choice = st.sidebar.radio("Select Model", ("Essential Features Model", "All Features Model"))
st.header("Input Bridge Data (Essential Only)")

# User inputs for essential features
Age = st.number_input("Age", min_value=0, max_value=200, value=30)
Span_ft = st.number_input("Span ft", min_value=0, max_value=600, value=300)
Deck_Width_ft = st.number_input("Deck Width ft", min_value=0, max_value=100, value=50)
Condition_Rating = st.number_input("Condition Rating", min_value=1, max_value=10, value=4)
Num_Lanes = st.number_input("Num Lanes", min_value=1, max_value=6, value=6)
Material = st.selectbox("Material", options=["Steel", "Composite", "Concrete"])

# When the user clicks the Predict button
if st.button("Predict Max Load Tons"):
    if model_choice == "Essential Features Model":
        # Build a DataFrame from the essential features only
        input_data = pd.DataFrame({
            'Age': [Age],
            'Span ft': [Span_ft],
            'Deck Width ft': [Deck_Width_ft],
            'Condition Rating': [Condition_Rating],
            'Num Lanes': [Num_Lanes],
            'Material': [Material]
        })

        # Debugging: Print expected vs actual columns
        expected_columns = preprocessor_selected.get_feature_names_out()
        st.write("Expected columns:", expected_columns)
        st.write("Input data columns:", input_data.columns.tolist())

        # Ensure input columns match the expected columns
        expected_columns_set = set(expected_columns)
        actual_columns_set = set(input_data.columns)

        missing_columns = expected_columns_set - actual_columns_set
        extra_columns = actual_columns_set - expected_columns_set

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
        if extra_columns:
            st.warning(f"Unexpected extra columns: {extra_columns}")

        # Rename columns if necessary
        column_mapping = {"Span_ft": "Span ft", "Deck_Width_ft": "Deck Width ft"}
        input_data.rename(columns=column_mapping, inplace=True)

        # Add missing columns with default values
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Or an appropriate default value

        # Preprocess input using the selected-features preprocessor
        processed_data = preprocessor_selected.transform(input_data)

        # Get prediction from the essential-features model
        prediction = model_selected.predict(processed_data)
        st.success(f"Predicted Max Load Tons (Essential Model): ${prediction[0][0]:,.2f}")

    else:
        default_all = pd.read_csv('default_all_features.csv', index_col=0)

        # Overwrite the essential features with user inputs
        default_all.loc[0, 'Age'] = Age
        default_all.loc[0, 'Span ft'] = Span_ft
        default_all.loc[0, 'Deck Width ft'] = Deck_Width_ft
        default_all.loc[0, 'Condition Rating'] = Condition_Rating
        default_all.loc[0, 'Num Lanes'] = Num_Lanes
        default_all.loc[0, 'Material'] = Material

        # Debugging: Print expected vs actual columns for the all-features model
        expected_columns = preprocessor_all.get_feature_names_out()
        st.write("Expected columns:", expected_columns)
        st.write("Input data columns:", default_all.columns.tolist())

        expected_columns_set = set(expected_columns)
        actual_columns_set = set(default_all.columns)

        missing_columns = expected_columns_set - actual_columns_set
        extra_columns = actual_columns_set - expected_columns_set

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
        if extra_columns:
            st.warning(f"Unexpected extra columns: {extra_columns}")

        # Add missing columns with default values
        for col in expected_columns:
            if col not in default_all.columns:
                default_all[col] = 0  # Or an appropriate default value

        processed_data = preprocessor_all.transform(default_all)
        prediction = model_all.predict(processed_data)
        st.success(f"Predicted Max Load Tons (All Features Model): ${prediction[0][0]:,.2f}")



