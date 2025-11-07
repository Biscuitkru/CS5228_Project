# Singapore HDB Resale Price Prediction

A machine learning project for predicting Singapore HDB (Housing & Development Board) resale prices using advanced regression techniques and ensemble methods.

**Course:** CS5228 - Knowledge Discovery and Data Mining  
**Institution:** National University of Singapore (NUS)  
**Best Model:** XGBoost with RMSE ~25,400

---

##  Overview

This project tackles the challenge of predicting HDB resale prices in Singapore using historical transaction data and auxiliary datasets (MRT stations, schools, shopping malls, etc.). The goal is to build an accurate regression model that can help buyers, sellers, and policymakers understand pricing dynamics in Singapore's public housing market.

**Key Objectives:**
- Perform comprehensive exploratory data analysis (EDA)
- Engineer meaningful features from raw data
- Compare multiple regression algorithms (linear, polynomial, tree-based, ensemble)
- Optimize hyperparameters for best performance
- Generate accurate predictions for unseen test data

---

##  Dataset

The dataset consists of HDB resale transactions with the following components:

### Main Dataset
- **Training Set:** `data/train.csv` (~70,000 transactions)
- **Test Set:** `data/test.csv` (~30,000 transactions)
- **Target Variable:** `resale_price` (in SGD)

### Features (20+ attributes)
- **Temporal:** `month`, `lease_commence_date`, derived `flat_age`
- **Location:** `town`, `street_name`, `block`, `storey_range`
- **Property:** `flat_type`, `flat_model`, `floor_area_sqm`
- **Lease:** `remaining_lease` (years/months)

### Auxiliary Data
Located in `data/auxiliary-data/`:
- `sg-mrt-stations.csv` - MRT/LRT station coordinates
- `sg-primary-schools.csv` - Primary school locations
- `sg-secondary-schools.csv` - Secondary school locations
- `sg-shopping-malls.csv` - Major shopping mall locations
- `sg-hdb-block-details.csv` - Detailed HDB block information
- `sg-gov-hawkers.csv` - Hawker center locations

---

##  Features

### Feature Engineering
- **Flat Age Calculation:** `flat_age = year - lease_commence_date`
- **Remaining Lease Parsing:** Extract years/months from string format
- **One-Hot Encoding:** Categorical variables (town, flat_type, flat_model)
- **Numerical Scaling:** StandardScaler for continuous features
- **Missing Value Imputation:** Median (numerical), mode (categorical)

### Data Preprocessing Pipeline
```python
ColumnTransformer(
    numerical: SimpleImputer -> StandardScaler
    categorical: SimpleImputer -> OneHotEncoder
)
```

---

##  Models Implemented

| Model | Algorithm | Validation RMSE | Training Time | Notes |
|-------|-----------|----------------|---------------|-------|
| **Linear Regression** | OLS | ~55,386 | Fast | Baseline model |
| **Polynomial Regression** | Ridge (degree=2) | ~45,000 | Moderate | Interaction terms only |
| **Decision Tree** | CART | ~42,635 | Fast | High variance |
| **Bagging Regressor** | Bootstrap aggregating | ~31,511 | Moderate | Reduces variance |
| **Random Forest** | Ensemble trees | ~29,869 | Slow | Feature importance |
| **XGBoost**  | Gradient boosting | **~25,434** | Moderate | **Best model** |

### Why XGBoost Won?
1.  Sequential learning from errors (boosting)
2.  Advanced regularization (L1/L2, max_depth, gamma)
3.  Column/row subsampling (reduces overfitting)
4.  Early stopping with validation monitoring
5.  Log-transformed target (Gaussian residuals)

---

##  Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for XGBoost acceleration

### Step 1: Clone Repository
bash
git clone https://github.com/Biscuitkru/CS5228_Project```

### Step 2: Create Virtual Environment
bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate


---

## Usage

### Running the Complete Pipeline

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook CS5228_Project.ipynb
   ```

2. **Execute Cells Sequentially**
   - **Cells 1-8:** Import libraries and load data
   - **Cells 9-61:** Exploratory Data Analysis (EDA)
   - **Cells 62-75:** Train all baseline models
   - **Cell 76:** Train XGBoost with hyperparameter tuning (40 trials)
   - **Cell 77:** Generate predictions and create `submission.csv`

---

##  Project Structure


CS5228_Project/
├── README.md                              # Project documentation
├── CS5228_Project.ipynb                   # Main Jupyter notebook
├── requirements.txt                       # Python dependencies
├── submission.csv                         # Final predictions
│
├── data/                                  # Datasets
│   ├── train.csv                         # Training data
│   ├── test.csv                          # Test data
│   ├── example-submission.csv            # Submission format
│   └── auxiliary-data/                   # Additional datasets
│       ├── sg-mrt-stations.csv
│       ├── sg-primary-schools.csv
│       ├── sg-secondary-schools.csv
│       ├── sg-shopping-malls.csv
│       ├── sg-hdb-block-details.csv
│       └── sg-gov-hawkers.csv
│
├── models/                                # Saved models
    ├── linear_regression.joblib
    ├── polynomial_regression_optimized.joblib
    ├── decision_tree.joblib
    ├── bagging.joblib
    └── xgb_best/                        # Best XGBoost model
        ├── xgb_model.joblib             # Preprocessor
        ├── xgb_model.json               # XGBoost booster
        └── best_summary.json            # Hyperparameters




### Key Performance Insights

1. **Best RMSE:** ~25,400 SGD (on validation set)
2. **Hyperparameter Search:** 40 random trials with early stopping
3. **Target Transformation:** Log-transform improved residuals by ~15%
4. **Early Stopping:** Prevents overfitting at ~1,847 trees
5. **GPU Acceleration:** 3x faster training with CUDA

---

##  Key Insights

### From Exploratory Data Analysis

1. **Temporal Trends:**
   - Resale prices increased ~40% from 2012-2024
   - Significant spike during COVID-19 pandemic (2020-2022)

2. **Location Matters:**
   - Central region (Bishan, Toa Payoh) commands 20-30% premium
   - Proximity to MRT stations correlates with +10-15% price

3. **Property Characteristics:**
   - `floor_area_sqm` is strongest predictor (Pearson r = 0.78)
   - Newer flats (lower `flat_age`) have higher prices
   - Higher floors (storey_range) command slight premium

4. **Flat Type Distribution:**
   - 4-ROOM flats are most common (~35% of transactions)
   - EXECUTIVE flats have highest average price
   - 1-ROOM flats rare and price volatile

### Model Learnings

1. **Linear models fail** due to non-linear relationships
2. **Tree-based ensembles excel** at capturing interactions
3. **XGBoost's sequential learning** beats Random Forest's parallel trees
4. **Regularization critical** to prevent overfitting (L1=0.15, L2=2.8)
5. **Log-transform essential** for handling outliers and price skewness

---

##  Technologies Used

- **Python 3.13:** Core programming language
- **Jupyter Notebook:** Interactive development environment
- **pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **scikit-learn:** ML algorithms and preprocessing
- **XGBoost:** Gradient boosting framework
- **Matplotlib & Seaborn:** Data visualization
- **tqdm:** Progress bars for long-running tasks
- **joblib:** Model serialization
