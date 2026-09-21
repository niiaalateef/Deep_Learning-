# 🏠 House Price Prediction Model

> **Machine Learning Project — Regression | Python | scikit-learn | Pandas**

An end-to-end machine learning project for predicting residential property prices from structured property characteristics such as living area, bedrooms, quality, year built, garage capacity, and neighborhood.

---

## 📌 Project Overview

The goal is to build a regression model that estimates the sale price of a residential property from its characteristics.

### Workflow

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Categorical feature encoding
- Train/test splitting
- Multiple regression algorithms
- Model evaluation
- 5-fold cross-validation
- Model selection
- Final model training
- Model export with `pickle`
- Example prediction
- Saving evaluation results

### Target

```text
SalePrice_USD
```

> **Important:** The dataset used here contains 30 educational/demo records. It is not a production housing dataset and should not be treated as representative of a real housing market.

---

# 🗂️ Recommended GitHub Structure

```text
house-price-prediction/
│
├── 📄 real-ml-coding.py
├── 📄 house_price_dataset.xlsx
├── 📄 house_price_model.pkl
├── 📄 model_evaluation.xlsx
├── 📄 README.md
│
└── 📁 images/
    ├── price_distribution.png
    ├── living_area_vs_price.png
    └── quality_vs_price.png
```

---

# 📊 Dataset

The dataset contains **30 observations** and **10 columns**.

| Feature | Description |
|---|---|
| `Id` | Unique property identifier |
| `LotArea_sqft` | Lot area in square feet |
| `OverallQual` | Overall quality rating |
| `YearBuilt` | Year the property was built |
| `GrLivArea_sqft` | Above-ground living area in square feet |
| `Bedrooms` | Number of bedrooms |
| `FullBath` | Number of full bathrooms |
| `GarageCars` | Garage capacity in cars |
| `Neighborhood` | Property neighborhood/category |
| `SalePrice_USD` | Target sale price in USD |

---

# 🧠 Machine Learning Approach

This is a **supervised machine learning regression problem** because the target is a continuous numerical value.

Three algorithms are compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

Evaluation uses:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-fold Cross-Validation RMSE

The final model is selected using the lowest cross-validation RMSE.

---

# 🔧 Technologies & Libraries

```text
Python
Pandas
NumPy
Matplotlib
scikit-learn
OpenPyXL
Pickle
```

### Installation

```bash
py -m pip install pandas numpy matplotlib scikit-learn openpyxl
```

If `py` is unavailable:

```bash
python -m pip install pandas numpy matplotlib scikit-learn openpyxl
```

---

# 🔄 Project Workflow

```text
Raw Dataset
     │
     ▼
Load Excel File
     │
     ▼
Data Inspection
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature / Target Separation
     │
     ▼
Train / Test Split
     │
     ▼
Preprocessing
 ┌───────────────┐
 │ Numeric Data  │ → Median Imputation
 │ Categorical   │ → One-Hot Encoding
 └───────────────┘
     │
     ▼
Train Multiple Models
     │
     ├── Linear Regression
     ├── Random Forest
     └── Gradient Boosting
     │
     ▼
Evaluate Models
     │
     ▼
5-Fold Cross-Validation
     │
     ▼
Select Model
     │
     ▼
Retrain on Full Dataset
     │
     ▼
Save house_price_model.pkl
     │
     ▼
Predict New Property
```

---

# 1️⃣ Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Purpose

| Library | Purpose |
|---|---|
| Pandas | Data loading and manipulation |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization |
| scikit-learn | Preprocessing, models, validation and metrics |
| Pickle | Saving the trained model |

---

# 2️⃣ Load the Dataset

```python
df = pd.read_excel("house_price_dataset.xlsx")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())
```

The dataset contains:

```text
30 rows
10 columns
```

---

# 3️⃣ Inspect the Dataset

```python
print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())
```

This checks:

- Data types
- Dataset dimensions
- Statistical characteristics
- Missing values

---

# 4️⃣ Exploratory Data Analysis

## Price Distribution

```python
plt.figure(figsize=(8, 5))
plt.hist(df["SalePrice_USD"], bins=10)
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price (USD)")
plt.ylabel("Frequency")
plt.show()
```

## Living Area vs Price

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["GrLivArea_sqft"], df["SalePrice_USD"])
plt.title("Living Area vs. Sale Price")
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Sale Price (USD)")
plt.show()
```

## Overall Quality vs Price

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["OverallQual"], df["SalePrice_USD"])
plt.title("Overall Quality vs. Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price (USD)")
plt.show()
```

These plots help explore relationships between important property variables and sale price.

---

# 5️⃣ Define Features and Target

```python
target = "SalePrice_USD"

X = df.drop(columns=[target, "Id"])
y = df[target]
```

The `Id` column is removed because it is an identifier rather than a meaningful predictive feature.

---

# 6️⃣ Identify Feature Types

```python
categorical_features = ["Neighborhood"]

numeric_features = [
    "LotArea_sqft",
    "OverallQual",
    "YearBuilt",
    "GrLivArea_sqft",
    "Bedrooms",
    "FullBath",
    "GarageCars"
]
```

`Neighborhood` is categorical, while the remaining input features are numerical.

---

# 7️⃣ Data Preprocessing

```python
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
```

### Why?

- Numerical missing values are replaced with the median.
- Categorical missing values are replaced with the most frequent category.
- `Neighborhood` is converted to numerical indicator columns using one-hot encoding.
- The preprocessing is kept inside the model pipeline for consistency.

---

# 8️⃣ Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The split is:

```text
80% → Training
20% → Testing
```

`random_state=42` makes the split reproducible.

---

# 9️⃣ Define Machine Learning Models

```python
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}
```

### Models

**Linear Regression** — models the target as a linear combination of features.

**Random Forest** — combines many decision trees to produce predictions.

**Gradient Boosting** — builds trees sequentially, with later trees improving previous errors.

---

# 🔟 Create Pipelines

```python
pipelines = {}

for name, model in models.items():

    pipelines[name] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
```

Each pipeline combines preprocessing and model training.

---

# 1️⃣1️⃣ Train and Evaluate

```python
results = []

for name, pipeline in pipelines.items():

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

print("\nModel Evaluation Results:")
print(results_df)
```

---

# 📏 Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual prices.

```text
Lower MAE = smaller average error
```

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more heavily.

```text
Lower RMSE = better
```

### R² — R-Squared

Measures the amount of variation in the target explained by the model.

A value closer to 1 indicates stronger explanatory performance on the evaluated data.

---

# 1️⃣2️⃣ 5-Fold Cross-Validation

```python
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, pipeline in pipelines.items():

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
    )

    cv_results.append({
        "Model": name,
        "CV_MAE": -scores["test_mae"].mean(),
        "CV_RMSE": -scores["test_rmse"].mean(),
        "CV_R2": scores["test_r2"].mean()
    })

cv_results_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results:")
print(cv_results_df)
```

### Why cross-validation?

Instead of depending on one train/test split, the model is evaluated across five validation folds. This gives a broader view of how consistently each model performs on the available data.

---

# 1️⃣3️⃣ Select the Final Model

```python
best_model_name = cv_results_df.loc[
    cv_results_df["CV_RMSE"].idxmin(),
    "Model"
]

print("\nSelected Model:", best_model_name)
```

The model with the **lowest cross-validation RMSE** is selected.

---

# 1️⃣4️⃣ Train the Final Model

```python
final_model = pipelines[best_model_name]

final_model.fit(X, y)
```

The selected pipeline is retrained using the complete available dataset.

---

# 1️⃣5️⃣ Save the Model

```python
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(final_model, file)

print("\nFinal model saved as house_price_model.pkl")
```

The trained pipeline is stored as:

```text
house_price_model.pkl
```

---

# 1️⃣6️⃣ Example Prediction

```python
new_house = pd.DataFrame({
    "LotArea_sqft": [8000],
    "OverallQual": [7],
    "YearBuilt": [2015],
    "GrLivArea_sqft": [2000],
    "Bedrooms": [3],
    "FullBath": [2],
    "GarageCars": [2],
    "Neighborhood": ["NAmes"]
})

predicted_price = final_model.predict(new_house)

print("\nPredicted House Price:")
print(predicted_price[0])
```

The model processes the new property and returns an estimated sale price.

---

# 1️⃣7️⃣ Save Evaluation Results

```python
with pd.ExcelWriter("model_evaluation.xlsx") as writer:

    results_df.to_excel(
        writer,
        sheet_name="Test Results",
        index=False
    )

    cv_results_df.to_excel(
        writer,
        sheet_name="Cross Validation",
        index=False
    )
```

This creates:

```text
model_evaluation.xlsx
```

with separate sheets for test performance and cross-validation performance.

---

# 🧩 Complete Python Code

The following is the consolidated implementation used for the project.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_excel("house_price_dataset.xlsx")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())


# ============================================================
# 2. DATA INSPECTION
# ============================================================

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(df["SalePrice_USD"], bins=10)
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price (USD)")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(
    df["GrLivArea_sqft"],
    df["SalePrice_USD"]
)
plt.title("Living Area vs. Sale Price")
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Sale Price (USD)")
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(
    df["OverallQual"],
    df["SalePrice_USD"]
)
plt.title("Overall Quality vs. Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price (USD)")
plt.show()


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

target = "SalePrice_USD"

X = df.drop(
    columns=[target, "Id"]
)

y = df[target]


# ============================================================
# 5. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
    "Neighborhood"
]

numeric_features = [
    "LotArea_sqft",
    "OverallQual",
    "YearBuilt",
    "GrLivArea_sqft",
    "Bedrooms",
    "FullBath",
    "GarageCars"
]


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )
}


# ============================================================
# 9. CREATE PIPELINES
# ============================================================

pipelines = {}

for name, model in models.items():

    pipelines[name] = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

for name, pipeline in pipelines.items():

    print(f"\nTraining {name}...")

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })


results_df = pd.DataFrame(
    results
)

print(
    "\nModel Evaluation Results:"
)

print(results_df)


# ============================================================
# 11. CROSS-VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_results = []


for name, pipeline in pipelines.items():

    scores = cross_validate(

        pipeline,

        X,

        y,

        cv=cv,

        scoring={

            "mae":
                "neg_mean_absolute_error",

            "rmse":
                "neg_root_mean_squared_error",

            "r2":
                "r2"
        }
    )


    cv_results.append({

        "Model": name,

        "CV_MAE":
            -scores["test_mae"].mean(),

        "CV_RMSE":
            -scores["test_rmse"].mean(),

        "CV_R2":
            scores["test_r2"].mean()
    })


cv_results_df = pd.DataFrame(
    cv_results
)


print(
    "\nCross-Validation Results:"
)

print(cv_results_df)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = cv_results_df.loc[

    cv_results_df[
        "CV_RMSE"
    ].idxmin(),

    "Model"
]


print(
    "\nSelected Model:",
    best_model_name
)


# ============================================================
# 13. TRAIN FINAL MODEL
# ============================================================

final_model = pipelines[
    best_model_name
]


final_model.fit(
    X,
    y
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

with open(
    "house_price_model.pkl",
    "wb"
) as file:

    pickle.dump(
        final_model,
        file
    )


print(
    "\nFinal model saved as "
    "house_price_model.pkl"
)


# ============================================================
# 15. EXAMPLE PREDICTION
# ============================================================

new_house = pd.DataFrame({

    "LotArea_sqft": [8000],

    "OverallQual": [7],

    "YearBuilt": [2015],

    "GrLivArea_sqft": [2000],

    "Bedrooms": [3],

    "FullBath": [2],

    "GarageCars": [2],

    "Neighborhood": ["NAmes"]
})


predicted_price = final_model.predict(
    new_house
)


print(
    "\nPredicted House Price:"
)

print(
    predicted_price[0]
)


# ============================================================
# 16. SAVE EVALUATION RESULTS
# ============================================================

with pd.ExcelWriter(
    "model_evaluation.xlsx"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Test Results",
        index=False
    )

    cv_results_df.to_excel(
        writer,
        sheet_name="Cross Validation",
        index=False
    )


print(
    "\nEvaluation results saved as "
    "model_evaluation.xlsx"
)


print(
    "\nProject completed successfully!"
)
```

---

# 📝 Key Concepts

### Regression

Regression is used when the target variable is continuous.

```text
House characteristics → Estimated house price
```

### Feature Selection

The project uses meaningful property characteristics such as:

```text
Lot Area
Overall Quality
Year Built
Living Area
Bedrooms
Bathrooms
Garage Capacity
Neighborhood
```

The `Id` column is excluded because it is an identifier.

### One-Hot Encoding

Categorical text such as `Neighborhood` is converted into numerical indicator columns so that machine learning algorithms can use it.

### Pipeline

A scikit-learn pipeline connects preprocessing and model training into one reusable workflow.

### Cross-Validation

5-fold cross-validation evaluates each model across multiple validation splits.

### Model Persistence

The trained model is saved as:

```text
house_price_model.pkl
```

so it can be loaded later without retraining.

---

# ⚠️ Dataset Limitation

This project uses a **small educational dataset containing 30 records**.

Therefore:

- It is not sufficient for production deployment.
- The reported metrics are primarily demonstrations of the ML workflow.
- The model should not be used for real financial or property decisions.
- A production implementation would require a much larger and representative dataset.
- External validation on genuinely unseen real-world data would be required before deployment.

---

# 🚀 Future Improvements

Possible extensions include:

- Larger real-world housing dataset
- More detailed location features
- Renovation and property-age features
- Distance to schools and public transport
- More advanced feature engineering
- Hyperparameter tuning
- XGBoost
- Outlier analysis
- Feature importance analysis
- Prediction intervals
- Model monitoring
- Automated retraining
- API deployment
- Dashboard integration

---

# 📚 Final Summary

This project demonstrates a complete machine learning workflow for residential house-price prediction:

```text
Data
 ↓
Exploration
 ↓
Preprocessing
 ↓
Train/Test Split
 ↓
Model Training
 ↓
Evaluation
 ↓
Cross-Validation
 ↓
Model Selection
 ↓
Final Training
 ↓
Model Saving
 ↓
Prediction
```

It provides practical experience with **Python, Pandas, NumPy, Matplotlib, and scikit-learn**, while demonstrating how a regression model can be prepared for reuse and deployment.

---

## 👩‍💻 Author

**Niaaa**

### Project

**House Price Prediction Model**

### Technologies

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `scikit-learn`

---

> ⭐ **GitHub tip:** Keep the full implementation in `real-ml-coding.py` and use this README to explain the project, methodology, and results. The dataset/model files can be included for demonstration; for larger production datasets and models, Git LFS or external storage may be more appropriate.
