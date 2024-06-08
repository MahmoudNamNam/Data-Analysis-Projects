from model import *



model = joblib.load('best_model.pkl')
test =load_data('./Data/test.csv')
apply_clustering
def preprocess_test_data(test, xlim, ylim):
    """Preprocess train, and, validation data."""
    test = filter_geographical_boundaries(test, xlim, ylim)
    test = apply_clustering(test)
    test = calculate_trip_distance(test)
    test = extract_datetime_features(test)
    drop_columns = ['vendor_id', 'passenger_count','id', 'pickup_datetime', 'store_and_fwd_flag', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    test.drop(columns=drop_columns, inplace=True, axis=1)

    return test


xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
test = preprocess_test_data(test,xlim, ylim)
feature_columns = ['month', 'season', 'weekday', 'is_weekend',
        'pickup_hour', 'pickup_cluster_label',
        'dropoff_cluster_label', 'trip_distance', 'bearing', 'is_rush_hour']


# Make predictions on the test set
test_predictions = model.predict(test[feature_columns])

# Prepare the submission file
submission = pd.DataFrame({
    'trip_duration': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file created.")
