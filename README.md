# F1 Race Predictor 2025

A machine learning-based Formula 1 race prediction system that uses historical race data, weather conditions, and race strategy to predict race outcomes for different Grand Prix events.

## Features

- **Advanced Data Collection**:
  - Historical race data from FastF1 API
  - Weather conditions and track temperature
  - Race strategy data (pit stops, safety cars, etc.)
  - Tyre compound usage and degradation
  - Driver and team performance metrics

- **Enhanced Machine Learning**:
  - Ensemble model combining Random Forest, Gradient Boosting, and XGBoost
  - Feature importance analysis
  - Cross-validation for model evaluation
  - Performance metrics (MAE, RMSE, R²)

- **Track-Specific Analysis**:
  - Circuit characteristics (speed, braking, DRS zones)
  - Weather impact on performance
  - Tyre strategy optimization
  - Safety car probability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/2025_f1_predictions.git
cd 2025_f1_predictions
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the prediction script:
```bash
python prediction4.py
```

2. The script will:
   - Load historical race data from the first three races of 2024
   - Process weather and strategy data
   - Train the ensemble model
   - Generate predictions for Monaco and Silverstone Grand Prix
   - Create visualization plots

3. Output includes:
   - Predicted race times for each driver
   - Model performance metrics
   - Feature importance analysis
   - Visualization plots saved as PNG files

## Model Features

The prediction model considers:
- Driver performance metrics
- Team performance
- Track characteristics
- Weather conditions
- Race strategy elements
- Tyre management
- Safety car probability

## Visualization

The script generates three-panel visualization plots:
1. Predicted vs. Qualifying Times
2. Feature Importance Analysis
3. Cross-validation Scores

## Dependencies

- fastf1
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost


