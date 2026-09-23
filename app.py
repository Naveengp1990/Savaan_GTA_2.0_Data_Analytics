import streamlit as st
import pandas as pd
import joblib

# IMPORTANT:
# This import is required because the saved model
# contains AddTotalAdsTransformer.
from custom_transformer import AddTotalAdsTransformer


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "advertising_model.pkl"
    )

    return model


model = load_model()


# ============================================================
# Application Header
# ============================================================

st.title("📊 Advertising Sales Prediction")

st.write(
    """
    Predict sales based on advertising expenditure
    across TV, Social Media, and Print channels.
    """
)

st.divider()


# ============================================================
# Input Section
# ============================================================

st.subheader("Enter Advertising Expenditure")


col1, col2 = st.columns(2)


with col1:

    tv_ads = st.number_input(
        "TV Advertising",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    social_media_ads = st.number_input(
        "Social Media Advertising",
        min_value=0.0,
        value=50.0,
        step=1.0
    )


with col2:

    print_ads = st.number_input(
        "Print Advertising",
        min_value=0.0,
        value=20.0,
        step=1.0
    )


# ============================================================
# Total Advertising
# ============================================================

total_ads = (
    tv_ads
    + social_media_ads
    + print_ads
)


st.info(
    f"Total Advertising Expenditure: {total_ads:.2f}"
)


# ============================================================
# Prediction
# ============================================================

if st.button(
    "Predict Sales",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "tv_ads": [tv_ads],
        "social_media_ads": [social_media_ads],
        "print_ads": [print_ads]
    })


    prediction = model.predict(
        input_data
    )


    predicted_sales = prediction[0]


    # ========================================================
    # Result
    # ========================================================

    st.success(
        "Prediction completed successfully!"
    )


    st.metric(
        "Predicted Sales",
        f"{predicted_sales:.2f}"
    )


    # ========================================================
    # Input Summary
    # ========================================================

    st.subheader("Input Summary")


    summary = pd.DataFrame({
        "Advertising Channel": [
            "TV",
            "Social Media",
            "Print",
            "Total"
        ],

        "Expenditure": [
            tv_ads,
            social_media_ads,
            print_ads,
            total_ads
        ]
    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Polynomial Regression with custom feature engineering"
)