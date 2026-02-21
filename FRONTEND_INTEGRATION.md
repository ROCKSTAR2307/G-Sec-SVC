# Frontend Integration Guide

## Overview
This guide explains how to integrate your React frontend with the Python backend API.

## Quick Start

### Step 1: Start the Backend Server

```bash
# From the backend repository root
pip install -r requirements.txt
python run.py
```

Backend will run on: `http://localhost:5000`

### Step 2: Configure Frontend API URL

In your React frontend, create an API client. Add this to your React app:

```javascript
// src/api/bondAPI.js

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const bondAPI = {
  // Train all models
  trainModels: async (bondType) => {
    const response = await fetch(`${API_BASE_URL}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bond_type: bondType })
    });
    return response.json();
  },

  // Get predictions from a model
  getPredictions: async (bondType, modelName) => {
    const response = await fetch(
      `${API_BASE_URL}/predictions/${bondType}/${modelName}`
    );
    return response.json();
  },

  // Get prediction chart as base64 PNG
  getChart: async (bondType, modelName) => {
    const response = await fetch(
      `${API_BASE_URL}/chart/${bondType}/${modelName}`
    );
    const data = await response.json();
    return data.chart; // Returns: "data:image/png;base64,..."
  },

  // Download predictions as CSV
  exportCSV: (bondType, modelName) => {
    window.location.href =
      `${API_BASE_URL}/export/${bondType}/${modelName}`;
  },

  // Get all model metrics
  getSummary: async (bondType) => {
    const response = await fetch(`${API_BASE_URL}/summary/${bondType}`);
    return response.json();
  }
};
```

### Step 3: Use in React Components

#### Example 1: Train Models on Demand

```javascript
import { bondAPI } from './api/bondAPI';
import { useState } from 'react';

function ModelTrainer() {
  const [isLoading, setIsLoading] = useState(false);
  const [bondType, setBondType] = useState('10yr');

  const handleTrain = async () => {
    setIsLoading(true);
    try {
      const result = await bondAPI.trainModels(bondType);
      console.log('Models trained:', result);
      alert('Models trained successfully!');
    } catch (error) {
      console.error('Training failed:', error);
      alert('Failed to train models');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <select value={bondType} onChange={(e) => setBondType(e.target.value)}>
        <option value="3yr">3-Year Bond</option>
        <option value="10yr">10-Year Bond</option>
      </select>
      <button onClick={handleTrain} disabled={isLoading}>
        {isLoading ? 'Training...' : 'Train Models'}
      </button>
    </div>
  );
}
```

#### Example 2: Display Predictions Chart

```javascript
import { bondAPI } from './api/bondAPI';
import { useState, useEffect } from 'react';

function PredictionsChart({ bondType, modelName }) {
  const [chartImage, setChartImage] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchChart = async () => {
      try {
        const chart = await bondAPI.getChart(bondType, modelName);
        setChartImage(chart);
      } catch (error) {
        console.error('Failed to fetch chart:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchChart();
  }, [bondType, modelName]);

  if (loading) return <div>Loading chart...</div>;
  if (!chartImage) return <div>Failed to load chart</div>;

  return (
    <div>
      <h2>{modelName.toUpperCase()}</h2>
      <img src={chartImage} alt={`${modelName} predictions`} />
    </div>
  );
}
```

#### Example 3: Display Model Metrics

```javascript
import { bondAPI } from './api/bondAPI';
import { useState, useEffect } from 'react';

function ModelMetrics({ bondType }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await bondAPI.getSummary(bondType);
        setMetrics(data.model_metrics);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [bondType]);

  if (loading) return <div>Loading metrics...</div>;

  return (
    <div>
      <h2>Model Performance ({bondType})</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>MAPE</th>
            <th>MAE</th>
            <th>MSE</th>
            <th>R²</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([modelName, m]) => (
            <tr key={modelName}>
              <td>{modelName}</td>
              <td>{m.mape.toFixed(4)}</td>
              <td>{m.mae.toFixed(4)}</td>
              <td>{m.mse.toFixed(4)}</td>
              <td>{m.r2.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

#### Example 4: Export Predictions

```javascript
import { bondAPI } from './api/bondAPI';

function ExportButton({ bondType, modelName }) {
  const handleExport = () => {
    bondAPI.exportCSV(bondType, modelName);
  };

  return (
    <button onClick={handleExport}>
      Download {modelName} Predictions (CSV)
    </button>
  );
}
```

#### Example 5: Full Predictions View

```javascript
import { bondAPI } from './api/bondAPI';
import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function PredictionsView({ bondType, modelName }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await bondAPI.getPredictions(bondType, modelName);

        // Format data for Recharts
        const chartData = result.predictions.dates.map((date, idx) => ({
          date: new Date(date).toLocaleDateString(),
          actual: result.predictions.actual[idx],
          predicted: result.predictions.predicted[idx]
        }));

        setData({
          metrics: result.metrics,
          chart: chartData
        });
      } catch (error) {
        console.error('Failed to fetch predictions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [bondType, modelName]);

  if (loading) return <div>Loading predictions...</div>;
  if (!data) return <div>Failed to load predictions</div>;

  return (
    <div>
      <h2>{modelName.toUpperCase()} Predictions ({bondType})</h2>

      {/* Metrics */}
      <div style={{ marginBottom: '20px' }}>
        <h3>Performance Metrics</h3>
        <p>MAPE: {data.metrics.mape.toFixed(4)}</p>
        <p>MAE: {data.metrics.mae.toFixed(4)}</p>
        <p>MSE: {data.metrics.mse.toFixed(4)}</p>
        <p>R²: {data.metrics.r2.toFixed(4)}</p>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data.chart}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="actual" stroke="#8884d8" />
          <Line type="monotone" dataKey="predicted" stroke="#82ca9d" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PredictionsView;
```

## Environment Variables

Create a `.env` file in your React app root:

```env
# Development
REACT_APP_API_URL=http://localhost:5000/api

# Production (when deployed)
# REACT_APP_API_URL=https://your-api-domain.com/api
```

## Data Flow

```
Frontend (React)
    ↓ (HTTP Request)
Backend API (Flask/Python)
    ↓ (Load CSV Data)
ML Engine
    ↓ (Train/Predict)
Return JSON/PNG/CSV
    ↓ (HTTP Response)
Frontend (Display Results)
```

## Typical Usage Flow

1. **User selects bond type** (3yr or 10yr)
2. **Frontend sends training request** → Backend trains all 4 models
3. **User selects a model** → Frontend fetches predictions
4. **Display options:**
   - Show prediction chart (PNG)
   - Show metrics table
   - Show prediction data
   - Export to CSV

## API Response Examples

### Training Response
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

### Predictions Response
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
    "actual": [102.5, 102.3, 102.1],
    "predicted": [102.48, 102.28, 102.12],
    "dates": ["2023-01-01", "2023-01-02", "2023-01-03"]
  }
}
```

## Troubleshooting

### CORS Errors
- Ensure backend is running on `http://localhost:5000`
- CORS is already enabled in the backend

### Image Not Loading
- Chart endpoint returns base64-encoded PNG
- Use directly in `<img src={chartData} />`

### CSV Export Issues
- Browser will handle the download automatically
- Check if pop-ups are blocked

### Slow First Request
- First request trains all models (takes 30-60 seconds)
- Subsequent requests are cached and instant

## Next Steps

1. Copy the API client code to your React project
2. Import and use `bondAPI` in your components
3. Test with `http://localhost:5000/api/health`
4. Build your UI around the data
5. Deploy both frontend and backend together

## Support

Refer to `BACKEND_API_DOCS.md` for detailed API documentation.
