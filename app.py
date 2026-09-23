# app.py
import streamlit as st
import pickle
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Ad Sales Predictor | IIT M Saavan", page_icon="📊", layout="centered")

# Header
st.title("📊 Advertising Sales Predictor")
st.markdown("Predict expected sales revenue based on your advertising budget allocation across TV, Social Media, and Print.")
st.markdown("---")

# Load the saved model pipeline
@st.cache_resource
def load_model():
    with open('advertising_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Sidebar Inputs (Strictly 3 variables as requested)
st.sidebar.header("💰 Input Advertising Budgets")
st.sidebar.markdown("Enter the budget allocated for each channel:")

tv_ads = st.sidebar.number_input("TV Ads Budget ($)", min_value=0.0, value=50.0, step=1.0)
social_media_ads = st.sidebar.number_input("Social Media Ads Budget ($)", min_value=0.0, value=50.0, step=1.0)
print_ads = st.sidebar.number_input("Print Ads Budget ($)", min_value=0.0, value=20.0, step=1.0)

# Prediction Section
st.markdown("---")
if st.button("🚀 Predict Sales", use_container_width=True):
    # Prepare input data exactly as the model expects (3 columns)
    input_data = pd.DataFrame({
        'tv_ads': [tv_ads],
        'social_media_ads': [social_media_ads],
        'print_ads': [print_ads]
    })
    
    # The pipeline automatically calculates 'total_ads_expenses' internally!
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.success(f"### Predicted Sales: {prediction:,.2f} units")
    st.balloons()
    
    # Show total budget context
    total_budget = tv_ads + social_media_ads + print_ads
    st.info(f"💡 *Total Ad Budget Entered: ${total_budget:,.2f}*")
