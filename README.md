# 📊 Advertising Sales Predictor | IIT M Saavan

Welcome to the **Advertising Sales Predictor**, developed for the Business Data Analytics student project at the **IIT M Saavan** technical event. 

This end-to-end machine learning application predicts product sales based on advertising budgets allocated across three primary channels: TV, Social Media, and Print.

## 🚀 Key Features
- **End-to-End ML Pipeline:** Bundles custom feature engineering, polynomial preprocessing, and Linear Regression into a single, leak-free pipeline.
- **Interactive Web UI:** Built with Streamlit for a seamless, user-friendly prediction experience.
- **Zero Data Leakage:** Rigorously tested to ensure no target variables are used during feature engineering.

## 📂 Dataset Details
- **Source:** `Advertising_Data.xlsx`
- **Shape:** 200 rows × 5 columns
- **Input Features:** `tv_ads`, `social_media_ads`, `print_ads`
- **Target Variable:** `sales`

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** (Web Framework)
- **Scikit-Learn** (Machine Learning & Custom Pipelines)
- **Pandas & NumPy** (Data Manipulation)

## 🏃‍♂️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-folder>
