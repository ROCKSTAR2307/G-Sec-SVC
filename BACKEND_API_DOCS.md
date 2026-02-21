# Bond Market Prediction Backend API Documentation

## Overview
This backend service provides machine learning model training and prediction endpoints for bond market analysis. It supports 4 models (Linear Regression, XGBoost, ARIMA, LSTM) and 2 bond types (3-year and 10-year).

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

```bash
# 1. Navigate to project directory
cd /path/to/project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python run.py
```

The backend will be available at `http://localhost:5000`

## API Endpoints

### 1. Health Check
**Endpoint:** `GET /api/health`

**Description:** Check if the backend service is running

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 2. Train Models
**Endpoint:** `POST /api/train`

**Description:** Train all ML models for a specific bond type

**Request Body:**
```json
{
  "bond_type": "10yr"
}
```

**Bond Types:**
- `"3yr"` - 3-Year Government Securities
- `"10yr"` - 10-Year Government Securities

**Response:**
```json
{
  "status": "success",
  "bond_type": "10yr",
  "models_trained": {
    "linear_regression": {
      "mape": 0.0114,
      "mae": 0.0817,
      "mse": 0.0077,
      "r2": 0.7743
    },
    "xgboost": {
      "mape": 0.0013,
      "mae": 0.0398,
      "mse": 0.0026,
      "r2": 0.9938
    },
    "arima": {
      "mape": 0.0362,
      "mae": 0.2551,
      "mse": 0.0968,
      "r2": -1.8212
    },
    "lstm": {
      "mape": 0.0062,
      "mae": 0.0450,
      "mse": 0.0035,
      "r2": 0.8894
    }
  }
}
```

---

### 3. Get Predictions
**Endpoint:** `GET /api/predictions/<bond_type>/<model_name>`

**Description:** Get predictions from a trained model

**Parameters:**
- `bond_type`: `"3yr"` or `"10yr"`
- `model_name`: `"linear_regression"`, `"xgboost"`, `"arima"`, or `"lstm"`

**Example Request:**
```
GET /api/predictions/10yr/xgboost
```

**Response:**
```json
{
  "bond_type": "10yr",
  "model": "xgboost",
  "metrics": {
    "mape": 0.0013,
    "mae": 0.0398,
    "mse": 0.0026,
    "r2": 0.9938
  },
  "predictions": {
    "actual": [102.5, 102.3, 102.1, ...],
    "predicted": [102.48, 102.28, 102.12, ...],
    "dates": ["2023-01-01", "2023-01-02", "2023-01-03", ...]
  }
}
```

---

### 4. Get Prediction Chart
**Endpoint:** `GET /api/chart/<bond_type>/<model_name>`

**Description:** Get a base64-encoded PNG chart of actual vs predicted values

**Parameters:**
- `bond_type`: `"3yr"` or `"10yr"`
- `model_name`: `"linear_regression"`, `"xgboost"`, `"arima"`, or `"lstm"`

**Example Request:**
```
GET /api/chart/10yr/xgboost
```

**Response:**
```json
{
  "bond_type": "10yr",
  "model": "xgboost",
  "chart": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
}
```

**Usage in Frontend:**
```javascript
// Display the chart in an <img> tag
<img src={response.chart} alt="Prediction Chart" />
```

---

### 5. Export Predictions to CSV
**Endpoint:** `GET /api/export/<bond_type>/<model_name>`

**Description:** Download predictions as a CSV file

**Parameters:**
- `bond_type`: `"3yr"` or `"10yr"`
- `model_name`: `"linear_regression"`, `"xgboost"`, `"arima"`, or `"lstm"`

**Example Request:**
```
GET /api/export/10yr/xgboost
```

**Response:** CSV file download with columns:
- Date
- Actual
- Predicted
- Error

---

### 6. Get Summary Metrics
**Endpoint:** `GET /api/summary/<bond_type>`

**Description:** Get summary metrics for all models

**Parameters:**
- `bond_type`: `"3yr"` or `"10yr"`

**Example Request:**
```
GET /api/summary/10yr
```

**Response:**
```json
{
  "bond_type": "10yr",
  "data_points": 2945,
  "model_metrics": {
    "linear_regression": {
      "mape": 0.0114,
      "mae": 0.0817,
      "mse": 0.0077,
      "r2": 0.7743
    },
    "xgboost": {
      "mape": 0.0013,
      "mae": 0.0398,
      "mse": 0.0026,
      "r2": 0.9938
    },
    "arima": {
      "mape": 0.0362,
      "mae": 0.2551,
      "mse": 0.0968,
      "r2": -1.8212
    },
    "lstm": {
      "mape": 0.0062,
      "mae": 0.0450,
      "mse": 0.0035,
      "r2": 0.8894
    }
  }
}
```

---

## Frontend Integration Example

### React Example using Fetch API

```javascript
// Train models
async function trainModels(bondType) {
  const response = await fetch('http://localhost:5000/api/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bond_type: bondType })
  });
  return await response.json();
}

// Get predictions
async function getPredictions(bondType, modelName) {
  const response = await fetch(
    `http://localhost:5000/api/predictions/${bondType}/${modelName}`
  );
  return await response.json();
}

// Get chart
async function getChart(bondType, modelName) {
  const response = await fetch(
    `http://localhost:5000/api/chart/${bondType}/${modelName}`
  );
  const data = await response.json();
  return data.chart; // Base64 encoded PNG
}

// Export CSV
function exportCSV(bondType, modelName) {
  window.location.href =
    `http://localhost:5000/api/export/${bondType}/${modelName}`;
}

// Get summary
async function getSummary(bondType) {
  const response = await fetch(`http://localhost:5000/api/summary/${bondType}`);
  return await response.json();
}
```

---

## Error Handling

All endpoints return error responses in the following format:

```json
{
  "error": "Error description message"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Endpoint not found
- `500` - Server error

---

## Model Information

### Linear Regression
- Fast training
- Good for understanding feature relationships
- Best for: Quick analysis, baseline comparisons

### XGBoost
- Gradient boosting ensemble method
- Best overall performance (R² ≈ 0.99)
- Best for: Production predictions, high accuracy

### ARIMA
- Time series forecasting
- Captures temporal patterns
- Best for: Short-term forecasts, trend analysis

### LSTM
- Deep learning neural network
- Captures long-term dependencies
- Best for: Complex patterns, sequence predictions

---

## Performance Tips

1. **First Request Delay:** The first request for a bond type will load the CSV data and train models. This may take 30-60 seconds depending on system resources.

2. **Model Caching:** Once trained, models are cached in memory for subsequent requests.

3. **Large Datasets:** The backend efficiently handles datasets with 2,000+ records.

4. **Concurrent Requests:** Use connection pooling for multiple simultaneous requests.

---

## Configuration

Edit `backend/config.py` to customize:
- Debug mode
- Upload folder location
- Data folder location

---

## Troubleshooting

**Issue:** Connection refused
- **Solution:** Ensure the server is running (`python run.py`)

**Issue:** CSV files not found
- **Solution:** Ensure `p_merged_data_3.csv` and `p_merged_data_10.csv` are in the project root

**Issue:** Model training takes too long
- **Solution:** This is normal for LSTM on first run. Subsequent requests will be faster.

**Issue:** CORS errors in frontend
- **Solution:** CORS is already enabled. Check that requests use the correct URL format.

---

## Support

For issues or questions, check the project documentation or review the backend logs for detailed error information.
