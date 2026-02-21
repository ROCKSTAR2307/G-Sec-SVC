# G-Sec Bond Market Prediction — Backend API Documentation

## Overview
This backend service exposes machine learning model training and prediction endpoints for G-Sec bond market analysis. 

NOTE: The service is entirely stateless. Every api call triggers fresh loading and live-training. There is no cache. Values represent real ML algorithm execution.

- **4 Models:** Linear Regression, XGBoost, ARIMA, LSTM
- **2 Bond Types:** 3-Year G-Sec (`3yr`) and 10-Year G-Sec (`10yr`)
- **Base URL:** `http://localhost:8000`

---

## Setup & Installation

```bash
# 1. Navigate to backend project directory
cd /path/to/G-Sec-Backend-SVC

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python run.py
```

> The backend will be available at **`http://localhost:8000`**

---

## ⭐ Primary Endpoint — Used by Frontend Prediction Screen

### `POST /api/compute`

**This is the ONLY endpoint the frontend needs to call.**

When the user selects a model + bond type and clicks **Compute**, call this endpoint.
- It automatically picks the correct CSV (`p_merged_data_3.csv` for `3yr`, `p_merged_data_10.csv` for `10yr`)
- Trains the model from scratch on the fly.
- Returns the 4 key metric values **AND the line-chart data (dates, actuals, predicted)**.

#### Request Body
```json
{
  "model": "xgboost",
  "bond_type": "3yr"
}
```

#### Parameters
| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `model` | string | `linear_regression`, `xgboost`, `arima`, `lstm` | The ML model to run |
| `bond_type` | string | `3yr`, `10yr` | Selects the 3-yr or 10-yr CSV dataset |

#### Success Response `200`
```json
{
  "model": "xgboost",
  "bond_type": "3yr",
  "metrics": {
    "mape": 0.001342,
    "mae":  0.039812,
    "mse":  0.002631,
    "r2":   0.993800
  },
  "chart_data": {
    "dates": [
        "2022-01-01",
        "2022-01-02"
    ],
    "actual": [
        100.23,
        100.86
    ],
    "predicted": [
        100.19,
        100.84
    ]
  }
}
```

#### The 4 Returned Values
| Key | Full Name | Description |
|-----|-----------|-------------|
| `mape` | Mean Absolute Percentage Error | Lower is better. Prediction accuracy as a % |
| `mae` | Mean Absolute Error | Lower is better. Average absolute error |
| `mse` | Mean Squared Error | Lower is better. Penalises large errors |
| `r2` | R-Squared Score | Closer to 1.0 is better. Fit quality (0–1) |

#### Error Response `400`
```json
{
  "error": "Invalid model. Choose one of: ['linear_regression', 'xgboost', 'arima', 'lstm']"
}
```

#### Frontend Usage (React)
```javascript
async function computeModel(model, bondType) {
  const response = await fetch('http://localhost:8000/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: model, bond_type: bondType })
  });
  const data = await response.json();
  
  // Plot these on a graph:
  // data.chart_data.dates
  // data.chart_data.actual
  // data.chart_data.predicted

  // Display these in a table:
  // data.metrics.mape, data.metrics.mae, data.metrics.mse, data.metrics.r2
  return data;
}

// Example: User selects XGBoost + 3-year, clicks Compute
computeModel('xgboost', '3yr').then(data => {
  console.log(data); 
});
```

---

## All Available Endpoints

### 1. Health Check
**`GET /api/health`**

Check if the server is running.

```json
{ "status": "healthy" }
```

---

## Model + Dataset Mapping

| Frontend Selection | `model` param | `bond_type` param | CSV Used |
|--------------------|--------------|-------------------|----------|
| Linear Regression + 3yr | `linear_regression` | `3yr` | `p_merged_data_3.csv` |
| Linear Regression + 10yr | `linear_regression` | `10yr` | `p_merged_data_10.csv` |
| XGBoost + 3yr | `xgboost` | `3yr` | `p_merged_data_3.csv` |
| XGBoost + 10yr | `xgboost` | `10yr` | `p_merged_data_10.csv` |
| ARIMA + 3yr | `arima` | `3yr` | `p_merged_data_3.csv` |
| ARIMA + 10yr | `arima` | `10yr` | `p_merged_data_10.csv` |
| LSTM + 3yr | `lstm` | `3yr` | `p_merged_data_3.csv` |
| LSTM + 10yr | `lstm` | `10yr` | `p_merged_data_10.csv` |

---

## Error Responses

All endpoints return errors in this format:

```json
{ "error": "Error description message" }
```

| Status | Meaning |
|--------|---------|
| `200` | Success |
| `400` | Bad request — invalid `model` or `bond_type` |
| `404` | Endpoint not found |
| `500` | Server error (check logs) |

---

## Performance Notes

| Note | Detail |
|------|--------|
| **Stateless** | Every click starts a live-training. There is no cache. |
| **LSTM is slowest** | LSTM training takes 30–60s due to deep learning |
| **Linear Regression** | Fastest — trains in under 1 second |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` | Run `python run.py` to start the server. Confirm running on port 8000 |
| `Data file not found` | Ensure `p_merged_data_3.csv` and `p_merged_data_10.csv` are in the project root |
| `CORS error` | CORS is already enabled. Verify the request URL uses `http://localhost:8000` |
