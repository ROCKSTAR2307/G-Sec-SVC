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

Backend will run on: `http://localhost:8000` (Port was changed to 8000 to avoid macOS conflicts).

### Step 2: Configure Frontend API Client

We have simplified the backend to make your frontend exactly one call. 

In your React frontend, create an API client. Add this to your React app:

```javascript
// src/api/bondAPI.js

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

export const bondAPI = {
  // Primary Endpoint: Trains the model and gets BOTH metrics and chart data in one call
  compute: async (bondType, modelName) => {
    const response = await fetch(`${API_BASE_URL}/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        model: modelName,
        bond_type: bondType 
      })
    });
    
    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  }
};
```

### Step 3: Use in React Components (Vibe Coding)

Here is a complete, copy-pasteable React component using `recharts` to plot the graph and display your metrics.

```javascript
import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { bondAPI } from './api/bondAPI'; // Adjust path if needed

function PredictionScreen() {
  const [bondType, setBondType] = useState('10yr');
  const [model, setModel] = useState('xgboost');
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const handleCompute = async () => {
    setIsLoading(true);
    setError(null);
    setData(null);
    
    try {
      // One call gets everything (Metrics + Chart Data)
      const result = await bondAPI.compute(bondType, model);
      
      // Format the data for Recharts (combining dates, actuals, predicted)
      const formattedChartData = result.chart_data.dates.map((dateStr, index) => {
        // You can format the date nicer here if you want
        const dateObj = new Date(dateStr);
        return {
          date: dateObj.toLocaleDateString(),
          Actual: result.chart_data.actual[index],
          Predicted: result.chart_data.predicted[index]
        };
      });

      setData({
        metrics: result.metrics,
        chartData: formattedChartData
      });
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>G-Sec Bond Prediction</h1>
      
      {/* --- Controls --- */}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
        <select value={bondType} onChange={e => setBondType(e.target.value)} disabled={isLoading}>
          <option value="3yr">3-Year Bond</option>
          <option value="10yr">10-Year Bond</option>
        </select>
        
        <select value={model} onChange={e => setModel(e.target.value)} disabled={isLoading}>
          <option value="xgboost">XGBoost</option>
          <option value="lstm">LSTM (Deep Learning)</option>
          <option value="linear_regression">Linear Regression</option>
          <option value="arima">ARIMA</option>
        </select>
        
        <button onClick={handleCompute} disabled={isLoading}>
          {isLoading ? 'Computing...' : 'Compute'}
        </button>
      </div>

      {error && <div style={{ color: 'red' }}>Error: {error}</div>}

      {/* --- Results --- */}
      {data && (
        <div>
          {/* Metrics Panel */}
          <div style={{ 
            display: 'flex', gap: '20px', marginBottom: '30px', 
            padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' 
          }}>
            <div><strong>MAPE:</strong> {data.metrics.mape.toFixed(4)}</div>
            <div><strong>MAE:</strong> {data.metrics.mae.toFixed(4)}</div>
            <div><strong>MSE:</strong> {data.metrics.mse.toFixed(4)}</div>
            <div><strong>R²:</strong> {data.metrics.r2.toFixed(4)}</div>
          </div>

          {/* Interactive Chart */}
          <div style={{ height: '400px', width: '100%' }}>
            <ResponsiveContainer>
              <LineChart data={data.chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="Actual" 
                  stroke="#2563eb" 
                  dot={false}
                  strokeWidth={2}
                />
                <Line 
                  type="monotone" 
                  dataKey="Predicted" 
                  stroke="#ef4444" 
                  strokeDasharray="5 5" 
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

export default PredictionScreen;
```

## Data Flow
The logic is completely stateless now.
```
Frontend (React) User clicks Compute
    ↓ (HTTP POST Request with model and bond_type)
Backend API
    ↓ Loads CSV freshly
    ↓ Runs Machine Learning Algorithm exactly matching your thesis algorithms
Returns JSON containing 4 Metrics + Raw Line Chart Array
    ↓ (HTTP Response)
Frontend (Plots graph using Recharts)
```

## Important Vibe Coding Notes
1. **No Caching:** The backend is completely stateless. Every time you hit `Compute`, it trains the model from scratch. This guarantees accurate, live results that match your thesis paper perfectly.
2. **LSTM Wait Times:** LSTM models are heavy deep-learning algorithms. They will take ~30-60 seconds to resolve. Add a loading spinner on your UI (like the `isLoading` state in the code above) so the user knows it's thinking.
3. **Recharts dependency:** Make sure your frontend has the charting library installed: `npm install recharts` (or use shadcn ui / tremor).
