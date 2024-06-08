import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def check_missing_data(train, validation):
    """Print the number of missing values in train, validation, and test data."""
    print(f'Checking for missing data in train: {train.isna().sum().sum()}')
    print(f'Checking for missing data in validation: {validation.isna().sum().sum()}')

def filter_geographical_boundaries(df, xlim, ylim):
    """Filter data within specified geographical boundaries."""
    return df[
        (df.pickup_longitude > xlim[0]) & (df.pickup_longitude < xlim[1]) & 
        (df.dropoff_longitude > xlim[0]) & (df.dropoff_longitude < xlim[1]) &
        (df.pickup_latitude > ylim[0]) & (df.pickup_latitude < ylim[1]) & 
        (df.dropoff_latitude > ylim[0]) & (df.dropoff_latitude < ylim[1])
    ]

def apply_clustering(df, n_clusters=5):
    """Apply KMeans clustering to pickup and dropoff coordinates."""
    pickup_coordinates = df[['pickup_latitude', 'pickup_longitude']]
    dropoff_coordinates = df[['dropoff_latitude', 'dropoff_longitude']]

    pickup_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['pickup_cluster_label'] = pickup_kmeans.fit_predict(pickup_coordinates)

    dropoff_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['dropoff_cluster_label'] = dropoff_kmeans.fit_predict(dropoff_coordinates)
    
    return df

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371 * c
    return km

def calculate_bearing(row):
    """Calculate bearing between two points."""
    lat1 = np.radians(row['pickup_latitude'])
    lat2 = np.radians(row['dropoff_latitude'])
    diff_long = np.radians(row['dropoff_longitude'] - row['pickup_longitude'])
    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def calculate_trip_distance(df):
    """Add trip distance calculated using the Haversine formula and bearing."""
    df['trip_distance'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)
    df['bearing'] = df.apply(calculate_bearing, axis=1)
    return df

def extract_datetime_features(df):
    """Extract features from datetime."""
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['month'] = df['pickup_datetime'].dt.month
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if 18 <= x <= 21 else 0)
    return df

def remove_outliers(df, column):
    """Remove outliers from a specified column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def preprocess_data(train, validation, xlim, ylim):
    """Preprocess train, and, validation data."""
    train = filter_geographical_boundaries(train, xlim, ylim)
    validation = filter_geographical_boundaries(validation, xlim, ylim)
    

    train = apply_clustering(train)
    validation = apply_clustering(validation)
    

    train = calculate_trip_distance(train)
    validation = calculate_trip_distance(validation)
    

    train = extract_datetime_features(train)
    validation = extract_datetime_features(validation)
    

    train = remove_outliers(train, 'trip_duration')
    validation = remove_outliers(validation, 'trip_duration')

    drop_columns = ['vendor_id', 'id', 'pickup_datetime', 'store_and_fwd_flag', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    train.drop(columns=drop_columns, inplace=True, axis=1)
    validation.drop(columns=drop_columns, inplace=True, axis=1)

    return train, validation

def train_model(X_train, y_train):
    """Train the Ridge regression model with GridSearchCV."""
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),  # Polynomial features
        ('scaler', StandardScaler()),    # Feature scaling
        ('ridge', Ridge())               # Ridge regression
    ])

    param_grid = {
        'poly__degree': [2, 3, 4],         # Degrees of polynomial features
        'ridge__alpha': [0.1, 1.0, 10.0]  # Ridge regularization parameter
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def evaluate_model(model, X, y):
    """Evaluate the model and print cross-validated RMSE."""
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print(f'Cross-validated RMSE: {cv_rmse.mean()} ± {cv_rmse.std()}')

def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")

def predict_and_save_submission(model, test_features, test_ids, filename):
    """Predict test data and save the submission file."""
    test_predictions = model.predict(test_features)
    submission = pd.DataFrame({
        'id': test_ids,
        'trip_duration': test_predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission saved as '{filename}'")

if __name__ == "__main__":
    train_data_path = './Data/train.csv'
    validation_data_path = './Data/val.csv'


    train = load_data(train_data_path)
    validation = load_data(validation_data_path)


    print(train.shape)  # (1000000, 10)
    print(validation.shape)  # (229319, 10)


    check_missing_data(train, validation)

    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]

    train, validation = preprocess_data(train, validation, xlim, ylim)

    feature_columns = [
        'month', 'season', 'weekday', 'is_weekend',
        'pickup_hour', 'pickup_cluster_label',
        'dropoff_cluster_label', 'trip_distance', 'bearing', 'is_rush_hour'
    ]


    X_train = train[feature_columns]

    y_train = train['trip_duration']
    X_val = validation[feature_columns]
    y_val = validation['trip_duration']

    # Train the model
    #* best_model = train_model(X_train, y_train)
    
    # Evaluate the model
    #* evaluate_model(best_model, X_val, y_val) #* Cross-validated RMSE: 270.97288971576575 ± 1.9641924925825047

    # Save the model
    # *save_model(best_model, 'best_model.pkl')

    # Load the model for validation
    loaded_model = joblib.load('best_model.pkl')
    # Predict on validation set
    y_val_pred = loaded_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f'Validation RMSE: {val_rmse}') #* Validation RMSE: 309
    print(f'Validation MAE: {val_mae}') #* Validation MAE: 230.8

