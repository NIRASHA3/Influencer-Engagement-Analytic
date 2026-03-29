# Influencer Engagement Analytic

Predict Instagram post engagement using machine learning. A FastAPI-powered application that estimates engagement metrics based on post characteristics and account statistics.

## ✨ Features

- **Dual ML Models**: XGBoost and Random Forest for engagement prediction
- **Interactive Web UI**: User-friendly interface for making predictions
- **Pre-trained Models**: Optimized on real Instagram analytics data
- **REST API**: Easy integration with other services
- **High Accuracy**: XGBoost model achieves 99.7% R² on test data

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository

```bash
git clone https://github.com/NIRASHA3/Influencer-Engagement-Analytic.git
cd Influencer-Engagement-Analytic/instagram-engagement-predictor
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
cd api
uvicorn app:app --reload
```

4. Open your browser
   - **UI**: [http://localhost:8000](http://localhost:8000)
   - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📊 How It Works

1. **Input Features**: Provide Instagram post metrics (likes, comments, shares, etc.)
2. **Model Selection**: Choose between XGBoost or Random Forest
3. **Prediction**: Get estimated engagement rate
4. **Results**: View prediction with confidence metrics

## 🏗️ Project Structure

```
.
├── instagram-engagement-predictor/
│   ├── api/                    # FastAPI backend & UI
│   ├── models/                 # Pre-trained ML models
│   ├── notebooks/              # ML training pipeline
│   ├── reports/                # Model metrics & analysis
│   └── requirements.txt
├── .gitignore
└── README.md (this file)
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI |
| **ML Models** | XGBoost, Random Forest |
| **Frontend** | HTML/CSS/JavaScript |
| **Data Processing** | Pandas, Scikit-learn |
| **Server** | Uvicorn |

## 📈 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **XGBoost** | 0.27 | 0.88 | **0.9970** |
| **Random Forest** | 2.61 | 6.77 | 0.8228 |

XGBoost is significantly more accurate and is the default model for predictions.

## 📚 API Endpoints

### Available Endpoints

- **`POST /predict`** - Default prediction (XGBoost)
- **`POST /predict/xgboost`** - XGBoost model prediction
- **`POST /predict/random_forest`** - Random Forest model prediction

### Request Body

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

### Response

```json
{
  "model": "xgboost",
  "predicted_engagement": 10.2461
}
```
