import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

def get_weather_data(session):
    """Extract weather data from session"""
    weather_data = session.weather_data
    if weather_data is not None:
        return {
            'AirTemp': weather_data['AirTemp'].mean(),
            'Humidity': weather_data['Humidity'].mean(),
            'Pressure': weather_data['Pressure'].mean(),
            'Rainfall': weather_data['Rainfall'].sum(),
            'TrackTemp': weather_data['TrackTemp'].mean(),
            'WindSpeed': weather_data['WindSpeed'].mean(),
            'WindDirection': weather_data['WindDirection'].mean()
        }
    return None

def get_strategy_data(session):
    """Extract race strategy data"""
    laps = session.laps
    try:
        pit_stops = len(laps[laps['PitInTime'].notna()])
    except:
        pit_stops = 0
        
    # Calculate tyre compound percentages
    if 'Compound' in laps.columns:
        compounds = laps['Compound'].value_counts()
        total_laps = len(laps)
        compound_percentages = {
            'SoftPercent': (compounds.get('SOFT', 0) / total_laps) * 100 if total_laps > 0 else 0,
            'MediumPercent': (compounds.get('MEDIUM', 0) / total_laps) * 100 if total_laps > 0 else 0,
            'HardPercent': (compounds.get('HARD', 0) / total_laps) * 100 if total_laps > 0 else 0
        }
    else:
        compound_percentages = {
            'SoftPercent': 0,
            'MediumPercent': 0,
            'HardPercent': 0
        }
        
    strategy_data = {
        'PitStops': pit_stops,
        'SafetyCarLaps': len(laps[laps['TrackStatus'] == 'SC']),
        'VirtualSafetyCarLaps': len(laps[laps['TrackStatus'] == 'VSC']),
        'RedFlagLaps': len(laps[laps['TrackStatus'] == 'Red']),
        'AverageTyreLife': laps['TyreLife'].mean() if 'TyreLife' in laps.columns else 0,
        **compound_percentages  # Add compound percentages
    }
    return strategy_data

def get_historical_data(year, race_number, session_type="R"):
    """Get historical race data with additional features"""
    session = fastf1.get_session(year, race_number, session_type)
    session.load()
    
    # Get lap times and telemetry
    laps_data = session.laps
    
    # Extract relevant features
    lap_features = []
    
    # Get weather and strategy data
    weather_data = get_weather_data(session)
    strategy_data = get_strategy_data(session)
    
    for idx, lap in laps_data.iterrows():
        if pd.isna(lap['LapTime']):
            continue
            
        # Get car telemetry for the lap
        tel = lap.get_telemetry()
        
        feature_dict = {
            'Driver': lap['Driver'],  # Store original driver code
            'LapTime': lap['LapTime'].total_seconds(),
            'Team': lap['Team'],
            'Position': lap['Position'],
            'Stint': lap['Stint'],
            'AverageSpeed': tel['Speed'].mean(),
            'MaxSpeed': tel['Speed'].max(),
            'AverageThrottle': tel['Throttle'].mean() if 'Throttle' in tel.columns else 0,
            'AverageBrake': tel['Brake'].mean() if 'Brake' in tel.columns else 0,
            'AverageRPM': tel['RPM'].mean() if 'RPM' in tel.columns else 0,
            'AverageDRS': tel['DRS'].mean() if 'DRS' in tel.columns else 0,
            'TyreLife': lap['TyreLife'] if 'TyreLife' in lap.index else 0,
            'Compound': lap['Compound'] if 'Compound' in lap.index else 'Unknown',
            'IsPersonalBest': lap['IsPersonalBest'] if 'IsPersonalBest' in lap.index else False,
            'TrackStatus': lap['TrackStatus'],
            'LapNumber': lap['LapNumber'],
            'RaceLapNumber': lap['LapNumber']  # Use LapNumber as RaceLapNumber
        }
        
        # Add weather data if available
        if weather_data:
            feature_dict.update(weather_data)
        
        # Add strategy data
        feature_dict.update(strategy_data)
        
        lap_features.append(feature_dict)
    
    return pd.DataFrame(lap_features)

def create_ensemble_model():
    """Create an ensemble of multiple models"""
    # Define base models
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Create voting regressor
    ensemble = VotingRegressor([
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb_model)
    ])
    
    return ensemble

def predict_race_winner(race_name, qualifying_data, historical_races=None):
    """
    Predict race winner for a specific Grand Prix with enhanced features
    
    Parameters:
    -----------
    race_name : str
        Name of the Grand Prix (e.g., "Monaco", "Silverstone")
    qualifying_data : dict
        Dictionary containing qualifying times and other data
    historical_races : list of tuples, optional
        List of (year, race_number) tuples for historical data
    """
    if historical_races is None:
        # Default to first 3 races of 2024
        historical_races = [(2024, i) for i in range(1, 4)]
    
    # Get historical data
    historical_data = pd.DataFrame()
    for year, race in historical_races:
        try:
            race_data = get_historical_data(year, race)
            historical_data = pd.concat([historical_data, race_data])
        except Exception as e:
            print(f"Could not load data for race {race}: {e}")
    
    # Create qualifying DataFrame
    qualifying_df = pd.DataFrame(qualifying_data)
    
    # Map full names to FastF1 3-letter codes
    driver_mapping = {
        "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER",
        "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB",
        "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS",
        "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO"
    }
    
    # Create reverse mapping for historical data
    reverse_driver_mapping = {v: k for k, v in driver_mapping.items()}
    
    # Convert driver codes in historical data to full names
    if not historical_data.empty:
        historical_data['FullName'] = historical_data['Driver'].map(reverse_driver_mapping)
        historical_data = historical_data[historical_data['FullName'].notna()]
        historical_data['Driver'] = historical_data['FullName']
        historical_data = historical_data.drop('FullName', axis=1)
    
    # Feature Engineering
    def prepare_features(df):
        """Prepare features for the model"""
        # Convert categorical variables to numerical
        df_encoded = pd.get_dummies(df, columns=['Team', 'Compound', 'TrackStatus'])
        
        # Drop unnecessary columns
        columns_to_drop = ['LapTime', 'Driver']
        df_encoded = df_encoded.drop(columns=[col for col in columns_to_drop if col in df_encoded.columns])
        
        return df_encoded
    
    # Prepare historical data
    historical_features = prepare_features(historical_data)
    qualifying_features = prepare_features(qualifying_df)
    
    # Ensure both datasets have the same columns
    common_columns = list(set(historical_features.columns) & set(qualifying_features.columns))
    X_historical = historical_features[common_columns]
    y_historical = historical_data['LapTime']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_historical, y_historical, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_qual_scaled = scaler.transform(qualifying_features[common_columns])
    
    # Create and train ensemble model
    model = create_ensemble_model()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_test = model.predict(X_test_scaled)
    predicted_times = model.predict(X_qual_scaled)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Driver': qualifying_df['Driver'],
        'Team': qualifying_df['Team'],
        'QualifyingTime': qualifying_df['QualifyingTime'],
        'PredictedRaceTime': predicted_times
    })
    
    # Sort by predicted race time
    results = results.sort_values('PredictedRaceTime')
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': common_columns,
        'Importance': model.named_estimators_['rf'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(15, 15))
    
    # Create subplots
    plt.subplot(3, 1, 1)
    plt.title(f'Predicted Race Times vs Qualifying Times - {race_name}')
    plt.scatter(results['QualifyingTime'], results['PredictedRaceTime'], alpha=0.5)
    plt.xlabel('Qualifying Time (s)')
    plt.ylabel('Predicted Race Time (s)')
    for i, txt in enumerate(results['Driver']):
        plt.annotate(txt, (results['QualifyingTime'].iloc[i], results['PredictedRaceTime'].iloc[i]))
    
    # Feature importance plot
    plt.subplot(3, 1, 2)
    plt.title('Top 10 Most Important Features')
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    
    # Cross-validation scores plot
    plt.subplot(3, 1, 3)
    plt.title('Cross-validation Scores')
    plt.boxplot(cv_scores)
    plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig(f'prediction_analysis_{race_name.lower().replace(" ", "_")}.png')
    
    # Print results
    print(f"\n🏁 Enhanced {race_name} Grand Prix Predictions 🏁\n")
    print(results.to_string(index=False))
    print(f"\n📊 Model Performance Metrics:")
    print(f"Mean Absolute Error: {mae:.2f} seconds")
    print(f"Root Mean Squared Error: {rmse:.2f} seconds")
    print(f"R² Score: {r2:.3f}")
    print(f"Cross-validation R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Print feature importance
    print("\n🔍 Top 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))
    
    return results

# Example usage for different races
if __name__ == "__main__":
    # Example qualifying data for Monaco GP with enhanced features
    monaco_qualifying = {
        "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", 
                  "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                  "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"],
        "QualifyingTime": [75.096, 75.180, 75.481, 75.546, 75.670,
                           75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5],
        "Team": ["McLaren", "McLaren", "Red Bull Racing", "Mercedes",
                 "RB", "Williams", "Ferrari", "Mercedes",
                 "Alpine", "Ferrari", "Aston Martin", "Aston Martin"],
        "Position": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "AverageSpeed": [180.0] * 12,  # Lower average speed for Monaco
        "MaxSpeed": [280.0] * 12,      # Lower max speed for Monaco
        "AverageThrottle": [70.0] * 12,  # Lower throttle usage for Monaco
        "AverageBrake": [30.0] * 12,     # Higher brake usage for Monaco
        "AverageRPM": [10000.0] * 12,    # Lower RPM for Monaco
        "AverageDRS": [0.1] * 12,        # Lower DRS usage for Monaco
        "Stint": [1] * 12,               # Initial stint
        "TyreLife": [0] * 12,            # New tyres
        "Compound": ["SOFT"] * 12,       # Starting compound
        "TrackStatus": ["Green"] * 12,   # Initial track status
        "LapNumber": [1] * 12,           # First lap
        "RaceLapNumber": [1] * 12,       # First race lap
        "AirTemp": [25.0] * 12,          # Expected air temperature
        "Humidity": [60.0] * 12,         # Expected humidity
        "Pressure": [1013.0] * 12,       # Expected pressure
        "Rainfall": [0.0] * 12,          # Expected rainfall
        "TrackTemp": [35.0] * 12,        # Expected track temperature
        "WindSpeed": [5.0] * 12,         # Expected wind speed
        "WindDirection": [180.0] * 12,   # Expected wind direction
        "PitStops": [2] * 12,            # Expected number of pit stops
        "SafetyCarLaps": [3] * 12,       # Expected safety car laps
        "VirtualSafetyCarLaps": [2] * 12,  # Expected VSC laps
        "RedFlagLaps": [0] * 12,         # Expected red flag laps
        "AverageTyreLife": [20] * 12,    # Expected average tyre life
        "SoftPercent": [40.0] * 12,      # Expected soft tyre usage
        "MediumPercent": [40.0] * 12,    # Expected medium tyre usage
        "HardPercent": [20.0] * 12       # Expected hard tyre usage
    }
    
    # Example qualifying data for Silverstone GP with enhanced features
    silverstone_qualifying = {
        "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", 
                  "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                  "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"],
        "QualifyingTime": [75.096, 75.180, 75.481, 75.546, 75.670,
                           75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5],
        "Team": ["McLaren", "McLaren", "Red Bull Racing", "Mercedes",
                 "RB", "Williams", "Ferrari", "Mercedes",
                 "Alpine", "Ferrari", "Aston Martin", "Aston Martin"],
        "Position": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "AverageSpeed": [250.0] * 12,  # Higher average speed for Silverstone
        "MaxSpeed": [350.0] * 12,      # Higher max speed for Silverstone
        "AverageThrottle": [85.0] * 12,  # Higher throttle usage for Silverstone
        "AverageBrake": [15.0] * 12,     # Lower brake usage for Silverstone
        "AverageRPM": [12000.0] * 12,    # Higher RPM for Silverstone
        "AverageDRS": [0.4] * 12,        # Higher DRS usage for Silverstone
        "Stint": [1] * 12,               # Initial stint
        "TyreLife": [0] * 12,            # New tyres
        "Compound": ["SOFT"] * 12,       # Starting compound
        "TrackStatus": ["Green"] * 12,   # Initial track status
        "LapNumber": [1] * 12,           # First lap
        "RaceLapNumber": [1] * 12,       # First race lap
        "AirTemp": [22.0] * 12,          # Expected air temperature
        "Humidity": [70.0] * 12,         # Expected humidity
        "Pressure": [1015.0] * 12,       # Expected pressure
        "Rainfall": [0.2] * 12,          # Expected rainfall
        "TrackTemp": [30.0] * 12,        # Expected track temperature
        "WindSpeed": [8.0] * 12,         # Expected wind speed
        "WindDirection": [270.0] * 12,   # Expected wind direction
        "PitStops": [1] * 12,            # Expected number of pit stops
        "SafetyCarLaps": [1] * 12,       # Expected safety car laps
        "VirtualSafetyCarLaps": [1] * 12,  # Expected VSC laps
        "RedFlagLaps": [0] * 12,         # Expected red flag laps
        "AverageTyreLife": [25] * 12,    # Expected average tyre life
        "SoftPercent": [30.0] * 12,      # Expected soft tyre usage
        "MediumPercent": [50.0] * 12,    # Expected medium tyre usage
        "HardPercent": [20.0] * 12       # Expected hard tyre usage
    }
    
    # Predict for Monaco GP
    print("\nPredicting Monaco Grand Prix...")
    monaco_results = predict_race_winner("Monaco", monaco_qualifying)
    
    # Predict for Silverstone GP
    print("\nPredicting British Grand Prix...")
    silverstone_results = predict_race_winner("Silverstone", silverstone_qualifying) 