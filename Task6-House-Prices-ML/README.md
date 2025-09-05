# 🏡 House Prices Prediction – Task 6

This project is part of **Task 6: Feature Engineering & Basic Modeling**.  
We build a **Linear Regression pipeline** to predict house sale prices.

---

## 📊 Steps Covered
1. **Feature Engineering**
   - Selected 5 numeric features: `OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath`
   - Encoded 2 categorical features: `Neighborhood, HouseStyle`
   - Scaled numeric features using `StandardScaler`.

2. **Model**
   - Linear Regression model.
   - Train-test split: 80%-20%.
   - Evaluation metrics: **MAE** and **R² Score**.

3. **Visualization**
   - Predicted vs Actual SalePrice plot.
   - Feature importance (coefficients) plot.

---

## 📂 Folder Structure
```
House-Prices-ML/
│── data/
│   └── cleaned_house_prices.csv
│── task6_feature_engineering_model.ipynb
│── README.md
│── requirements.txt
```

---

## 🛠 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🚀 Results
- Mean Absolute Error (MAE): ~20000 (varies per dataset)
- R² Score: ~0.80+
- Strong relationship between features & SalePrice.

---

## 📸 Visualizations
- **Predicted vs Actual SalePrice**
- **Top Feature Importances**

---

✅ Task 6 completed successfully.
