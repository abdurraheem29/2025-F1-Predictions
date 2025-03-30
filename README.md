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

## Results Analysis

### Model Performance
- **Accuracy Metrics**:
  - Mean Absolute Error (MAE): 0.33 seconds
  - Root Mean Squared Error (RMSE): 0.68 seconds
  - R² Score: 0.994
  - Cross-validation R² Score: 0.984 (±0.018)

### Feature Importance
1. Average Speed (44.6%)
2. Pressure (14.9%)
3. Hard Tyre Percentage (14.7%)
4. Track Temperature (8.6%)
5. Wind Direction (7.8%)

### Prediction Analysis

#### Monaco Grand Prix
- **Top 3 Predicted**:
  1. Lando Norris (McLaren)
  2. Oscar Piastri (McLaren)
  3. Max Verstappen (Red Bull Racing)
- **Key Insights**:
  - McLaren shows strong performance
  - Qualifying position has significant impact
  - Smaller time gaps between drivers

#### Silverstone Grand Prix
- **Top 3 Predicted**:
  1. Lando Norris (McLaren)
  2. George Russell (Mercedes)
  3. Yuki Tsunoda (RB)
- **Key Insights**:
  - More mixed order due to overtaking opportunities
  - Higher average speeds impact predictions
  - Different team performance characteristics

### Model Strengths
1. **High Accuracy**: R² score of 0.994 indicates excellent prediction capability
2. **Consistent Performance**: Low MAE of 0.33 seconds across different tracks
3. **Feature Understanding**: Clear identification of important factors
4. **Track Adaptation**: Different predictions for different circuit types

### Areas for Improvement
1. **Data Collection**:
   - Include more historical races
   - Add driver experience metrics
   - Incorporate team development data
   - Include car setup information

2. **Feature Engineering**:
   - Add track-specific features
   - Include driver-team chemistry metrics
   - Consider race weekend progression
   - Add historical performance at specific circuits

3. **Model Enhancement**:
   - Implement time series analysis
   - Add deep learning components
   - Include probabilistic predictions
   - Develop track-specific models

4. **Validation**:
   - Add more cross-validation folds
   - Implement out-of-sample testing
   - Include uncertainty estimates
   - Add confidence intervals

## Future Development
1. **Short-term Improvements**:
   - Add more races to historical data
   - Implement real-time weather updates
   - Add driver form analysis
   - Include team strategy predictions

2. **Long-term Goals**:
   - Develop race weekend simulation
   - Add championship points prediction
   - Implement driver transfer impact analysis
   - Create interactive visualization dashboard




