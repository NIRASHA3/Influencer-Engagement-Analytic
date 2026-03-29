# Instagram Engagement Predictor

FastAPI + HTML app to estimate Instagram post engagement using trained ML models.

## Project Structure

```
instagram-engagement-predictor/
├── api/
│   ├── app.py
│   ├── static/
│   │   ├── index.html
│   │   └── assets/
├── models/
│   ├── final_model_xgboost_all_features.pkl
│   ├── best_model_xgboost.pkl
│   └── final_model_random_forest_all_features.pkl
├── notebooks/
├── reports/
├── requirements.txt
└── .gitignore
```

## Features

- FastAPI backend with two prediction modes: XGBoost and Random Forest
- Single-page UI for feature input and prediction display
- Model endpoint routing from UI buttons to `/predict/{model_name}`
- Robust model path resolution across `api/`, `models/`, and project root

## How The Pipeline Connects

1. UI form in `api/static/index.html` collects feature values.
2. On submit, frontend sends JSON to `POST /predict/{model_name}`.
3. `api/app.py` validates payload with `InstaInput`.
4. Backend constructs feature dataframe in training-column order.
5. Selected model predicts and API returns `predicted_engagement`.
6. UI renders success/error message in the result panel.

## ML Pipeline Details

The training workflow in `notebooks/Influencer_Engagement_Analytic.ipynb` is built to make predictions stable and robust on skewed social media data.

### 1. Data Preparation

- Original training source: `Instagram_Post_Analytics.csv` (loaded in the notebook environment)
- Target variable: `engagement_rate`
- Basic checks: missing values and duplicates
- Outlier handling:
  - Percentile capping on `share_rate`, `save_rate`, and `engagement_rate` (0.5th and 99.5th percentile)
  - Additional cap: `engagement_rate <= 100`

### 2. Feature Split

- `X`: all columns except `engagement_rate`
- `y`: `engagement_rate`
- Train/test split: 80/20 with `random_state=42`

### 3. Preprocessing

- Numeric columns with high skew (`skew > 1`): `log1p` then `StandardScaler`
- Remaining numeric columns: `StandardScaler`
- Categorical columns: `OneHotEncoder(handle_unknown='ignore')`
- Implemented using a `ColumnTransformer` inside model pipelines

### 4. Modeling Strategy

- Models:
  - `RandomForestRegressor`
  - `XGBRegressor`
- Both wrapped with `TransformedTargetRegressor`:
  - target transform: `log1p(y)`
  - inverse transform: `expm1(y_pred)`
- Hyperparameter tuning with `GridSearchCV`
- Cross-validation with `RepeatedKFold`

### 5. Saved Artifacts

- `models/final_model_xgboost_all_features.pkl`
- `models/final_model_random_forest_all_features.pkl`
- `models/best_model_xgboost.pkl`
- `reports/model_comparison_summary.json`
- `reports/feature_importance.csv`

### 6. Holdout Metrics (Saved)

From `reports/model_comparison_summary.json`:

- Random Forest: `MAE = 2.6065`, `RMSE = 6.7688`, `R2 = 0.8228`
- XGBoost: `MAE = 0.2663`, `RMSE = 0.8805`, `R2 = 0.9970`

This indicates XGBoost is the stronger default model for this dataset.

## Requirements

Install from project root:

```bash
pip install -r requirements.txt
```

## Run The API + UI

From project root:

```bash
cd api
uvicorn app:app --reload
```

Then open:

- UI: http://127.0.0.1:8000/
- Swagger docs: http://127.0.0.1:8000/docs

## API Endpoints

- `POST /predict` (default model: xgboost)
- `POST /predict/xgboost`
- `POST /predict/random_forest`

Example request body:

```json
{
  "media_type": "Carousel",
  "likes": 12000,
  "comments": 2000,
  "shares": 1000,
  "saves": 3000,
  "reach": 200000,
  "impressions": 250000,
  "caption_length": 120,
  "hashtags_count": 15,
  "followers_gained": 50,
  "traffic_source": "Explore",
  "content_category": "Fitness",
  "share_rate": 0.4,
  "save_rate": 1.2,
  "engagement_score": 56000
}
```

Example response:

```json
{
  "model": "xgboost",
  "predicted_engagement": 10.2461
}
```

## Sample Test Cases (Dataset Row Validation)

The following rows were manually checked against the saved pipeline models to verify behavior.

### Sample 1: IG0000001

- Expected `engagement_rate`: `4.97`
- XGBoost prediction: `5.0153` (absolute error: `0.0453`)
- Random Forest prediction: `5.8868` (absolute error: `0.9168`)

### Sample 2: IG0000015

- Expected `engagement_rate`: `9.24`
- XGBoost prediction: `8.9221` (absolute error: `0.3179`)
- Random Forest prediction: `9.0288` (absolute error: `0.2112`)

### Sample 3: IG0000114

- Expected `engagement_rate`: `19.27`
- XGBoost prediction: `19.4233` (absolute error: `0.1533`)
- Random Forest prediction: `16.6192` (absolute error: `2.6508`)

These spot checks confirm API connectivity and model inference are working correctly. Per-row accuracy can vary, so evaluate using holdout metrics (`MAE`, `RMSE`) rather than only single-row differences.

## Reproduce A Sample Test Quickly

After starting the API (`cd api` then `uvicorn app:app --reload`), call:

```bash
curl -X POST "http://127.0.0.1:8000/predict/xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "media_type": "Reel",
    "likes": 29834,
    "comments": 3242,
    "shares": 1434,
    "saves": 377,
    "reach": 119869,
    "impressions": 377581,
    "caption_length": 539,
    "hashtags_count": 10,
    "followers_gained": 808,
    "traffic_source": "Home Feed",
    "content_category": "Lifestyle",
    "share_rate": 0.01196305967,
    "save_rate": 0.0031451,
    "engagement_score": 42128
  }'
```

## Notes

- The backend includes compatibility placeholders (`manual_sum`, `diff`) set to `0` to match trained feature shape.
- If you rename or replace model files, keep one valid model file per algorithm in `models/`.
