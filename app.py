import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras

# === Load pre-trained models and preprocessors ===
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')

# === Define default values for missing features ===
default_values = {
    'Age': 30,  # Overwritten by user input
    'Span ft': 300,  # Overwritten by user input
    'Deck Width ft': 20,  # Overwritten by user input
    'Condition Rating': 4,
    'Num Lanes': 6,  # Overwritten by user input
    'Material_Composite': 0,
    'Material_Concrete': 0,
    'Material_Steel': 1,  # Default material is Steel
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

# Function to preprocess input data
def preprocess_input(input_data, preprocessor):
    expected_columns = preprocessor.feature_names_in_

    # One-hot encode "Material" manually to match training data
    for mat in ["Material_Composite", "Material_Concrete", "Material_Steel"]:
        input_data[mat] = 0
    input_data[f"Material_{Material}"] = 1  # Set the correct material

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = default_values.get(col, 0)

    # Reorder columns to match the preprocessor's expectations
    input_data = input_data[expected_columns]

    # Debugging: Print expected vs actual columns
    st.write("Expected columns:", expected_columns)
    st.write("Final input data before preprocessing:", input_data)

    return preprocessor.transform(input_data)

# When the user clicks the Predict button
if st.button("Predict Max Load Tons"):
    if model_choice == "Essential Features Model":
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame({
            'Age': [Age],
            'Span ft': [Span_ft],
            'Deck Width ft': [Deck_Width_ft],
            'Condition Rating': [Condition_Rating],
            'Num Lanes': [Num_Lanes]
        })

        # Preprocess input
        processed_data = preprocess_input(input_data, preprocessor_selected)

        # Get prediction
        prediction = model_selected.predict(processed_data)
        st.success(f"Predicted Max Load Tons (Essential Model): {prediction[0][0]:,.2f}")

    else:
        # Load default dataset for full model
        default_all = pd.read_csv('default_all_features.csv', index_col=0)

        # Update essential features with user inputs
        default_all.loc[0, 'Age'] = Age
        default_all.loc[0, 'Span ft'] = Span_ft
        default_all.loc[0, 'Deck Width ft'] = Deck_Width_ft
        default_all.loc[0, 'Condition Rating'] = Condition_Rating
        default_all.loc[0, 'Num Lanes'] = Num_Lanes

        # Preprocess input
        processed_data = preprocess_input(default_all, preprocessor_all)

        # Get prediction
        prediction = model_all.predict(processed_data)
        st.success(f"Predicted Max Load Tons (All Features Model): {prediction[0][0]:,.2f}")




