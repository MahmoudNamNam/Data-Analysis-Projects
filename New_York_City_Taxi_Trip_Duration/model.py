import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def check_missing_data(train, validation):
    """Print the number of missing values in train and validation data."""
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

def preprocess_data(df, xlim, ylim):
    """Preprocess data."""
    df = filter_geographical_boundaries(df, xlim, ylim)
    df = apply_clustering(df)
    df = calculate_trip_distance(df)
    df = extract_datetime_features(df)
    drop_columns = ['vendor_id',  'pickup_datetime', 'store_and_fwd_flag', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    df.drop(columns=drop_columns, inplace=True, axis=1)
    return df

def train_model(X_train, y_train):
    """Train the Ridge regression model with GridSearchCV."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),    # Feature scaling
        ('poly', PolynomialFeatures()),  # Polynomial features
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

def load_model(filename):
    """Load a model from a file."""
    return joblib.load(filename)

def predict_and_save_submission(model, test_features, test_ids, filename):
    """Predict test data and save the submission file."""
    test_predictions = model.predict(test_features)
    submission = pd.DataFrame({
        'id': test_ids,
        'trip_duration': test_predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission saved as '{filename}'")

def main():
    train_data_path = './Data/train.csv'
    validation_data_path = './Data/val.csv'
    test_data_path = './Data/test.csv'
    sample_submission_path = './Data/sample_submission.csv'
    model_filename = 'model.pkl'
    submission_filename = 'submission.csv'

    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]

    action = input("Enter action (train, evaluate, predict): ").strip().lower()

    if action == 'train':
        train = load_data(train_data_path)
        validation = load_data(validation_data_path)

        check_missing_data(train, validation)

        train = preprocess_data(train, xlim, ylim)
        validation = preprocess_data(validation, xlim, ylim)

        train = remove_outliers(train, 'trip_duration')
        validation = remove_outliers(validation, 'trip_duration')

        feature_columns = [
            'month', 'season', 'weekday', 'is_weekend',
            'pickup_hour', 'pickup_cluster_label',
            'dropoff_cluster_label', 'trip_distance', 'bearing', 'is_rush_hour'
        ]

        X_train = train[feature_columns]
        y_train = train['trip_duration']

        best_model = train_model(X_train, y_train)
        evaluate_model(best_model, X_train, y_train)
        save_model(best_model, model_filename)

    elif action == 'evaluate':
        validation = load_data(validation_data_path)
        validation = preprocess_data(validation, xlim, ylim)
        validation = remove_outliers(validation, 'trip_duration')

        feature_columns = [
            'month', 'season', 'weekday', 'is_weekend',
            'pickup_hour', 'pickup_cluster_label',
            'dropoff_cluster_label', 'trip_distance', 'bearing', 'is_rush_hour'
        ]

        X_val = validation[feature_columns]
        y_val = validation['trip_duration']

        model = load_model(model_filename)
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        print(f'Validation RMSE: {val_rmse}')
        print(f'Validation MAE: {val_mae}')

    elif action == 'predict':
        test = load_data(test_data_path)
        sample_submission = load_data(sample_submission_path)

        test = preprocess_data(test, xlim, ylim)
        test_ids = test['id']
        feature_columns = [
            'month', 'season', 'weekday', 'is_weekend',
            'pickup_hour', 'pickup_cluster_label',
            'dropoff_cluster_label', 'trip_distance', 'bearing', 'is_rush_hour'
        ]
        test_features = test[feature_columns]

        model = load_model(model_filename)
        y_test_pred = model.predict(test_features)
        
        predict_and_save_submission(model, test_features, test_ids, submission_filename)

        if 'trip_duration' in sample_submission.columns:
            sample_submission.set_index('id', inplace=True)
            try:
                y_sample_true = sample_submission.loc[test_ids]['trip_duration']
                test_rmse = np.sqrt(mean_squared_error(y_sample_true, y_test_pred))
                test_mae = mean_absolute_error(y_sample_true, y_test_pred)
                print(f'Sample Submission RMSE: {test_rmse}')
                print(f'Sample Submission MAE: {test_mae}')
            except KeyError as e:
                print(f"KeyError: {e}. Ensure that test_ids are present in the sample submission.")
        else:
            print("Column 'trip_duration' not found in sample_submission.")

if __name__ == "__main__":
    main()
