# 🏠 Real Estate Investment Advisor

An end-to-end machine learning and visual analytics web application built with **Streamlit**, **Scikit-Learn**, and **LightGBM** to evaluate Indian residential properties for investment suitability and forecast 5-year capital appreciation.

🔗 **Live Demo:** [Real Estate Investment Advisor on Streamlit Cloud](https://real-estate-investment-advisor-rpyeezqfxjw66646agx5qn.streamlit.app)

---

## 📌 Overview

Evaluating residential real estate involves balancing location demand, per-square-foot pricing, property age, furnishing, and surrounding civic infrastructure. **Real Estate Investment Advisor** replaces subjective property estimates with a dual-model ML pipeline trained on Indian housing market data:

1. **Investment Classification (`Good_Investment`):** Predicts whether a property represents a financially sound investment (`1` = Good Investment, `0` = Not Recommended) alongside an **Investment Suitability Potential (Match Score %)**.
2. **5-Year Valuation Regression (`Future_Price_5Y`):** Forecasts expected property market value five years into the future (in Lakhs ₹), quantifying projected capital gain and percentage appreciation.

---

## ✨ Key Features

* **Live Investment Predictor (Tab 1):**
  * Interactive 3-column input form covering **Location & Type** (cascading City $\rightarrow$ Locality and State filtering), **Size & Pricing** (BHK, SqFt, current price with live `₹ / SqFt` rate calculation, property age), and **Infrastructure & Amenities** (schools, hospitals, public transport sliders, parking, facing direction, owner type, availability).
  * Instant investment recommendation card, match score progress bar, 5-year price forecast, and a **Key Valuation Summary** comparing the property's rate against the local median price per SqFt.
* **Market Analytics Dashboard (Tab 2):**
  * Exploratory visualizations powered by **Matplotlib** and **Seaborn**:
    * *Average Price in Lakhs by City* (Horizontal Bar Chart)
    * *Price per SqFt Distribution by Property Type* (Box Plot)
    * *Transport Accessibility vs. Investment Potential* (Grouped Count Plot)
    * *Infrastructure Score vs. Price per SqFt* (Scatter Plot)
* **Model Performance & Architecture (Tab 3):**
  * Transparent breakdown of the preprocessing pipeline, MLflow evaluation metrics, and live inspection of loaded `LGBMClassifier` and `LGBMRegressor` hyperparameters.

---

## 🧠 Machine Learning Architecture

* **Preprocessing (`ColumnTransformer`):** Encapsulates `StandardScaler` for numerical features and `OneHotEncoder` for categorical variables inside Scikit-Learn `Pipeline` objects to eliminate training-serving skew.
* **Models (`LightGBM`):**
  * `LGBMClassifier(max_depth=6, random_state=42, verbose=-1)` for binary classification and probability scoring (`predict_proba`).
  * `LGBMRegressor(max_depth=6, random_state=42, verbose=-1)` for 5-year valuation forecasting.
  * Benchmarked against `RandomForestClassifier` and `RandomForestRegressor`.
* **Experiment Tracking (`MLflow`):** Logs hyperparameters, model artifacts (`.skops`), classification metrics (**Accuracy, F1-Score, ROC-AUC**), and regression metrics (**RMSE, MAE, $R^2$**).

---

## 📂 Repository Structure

```text
Real-Estate-Investment-Advisor/
├── app.py                                      # Main Streamlit web application
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── Data/
│   ├── india_housing_prices.csv                # Raw housing dataset
│   └── processed/
│       └── cleaned_housing_prices.csv          # Cleaned dataset used by the app
├── models/
│   ├── best_classification_pipeline.pkl        # Serialized LightGBM classification pipeline
│   └── best_regression_pipeline.pkl            # Serialized LightGBM regression pipeline
└── Notebooks/
    ├── real_state_advisor.ipynb                # EDA, preprocessing, model training & MLflow logging
    └── mlruns/                                 # MLflow experiment tracking artifacts
```

---

## 🚀 Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/adityaa07singh/Real-Estate-Investment-Advisor.git](https://github.com/adityaa07singh/Real-Estate-Investment-Advisor.git)
   cd Real-Estate-Investment-Advisor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note for macOS users: LightGBM requires OpenMP. Install it via Homebrew with `brew install libomp` if needed.)*

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 👨‍💻 Author

**Aditya Singh**  
Master of Computer Applications (MCA)  
Department of Information Technology and Computer Application  
Madan Mohan Malaviya University of Technology (MMMUT), Gorakhpur  
*Supervised by Dr. Preeti Singh*
