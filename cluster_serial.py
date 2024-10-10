import pandas as pd
from sklearn.cluster import KMeans
import time

# Load the CSV file 
df = pd.read_csv('synthetic_large_clustering_data.csv')

# Convert the dataframe into a NumPy array for use with KMeans
X_serial = df.values

# Initialize the KMeans model
n_clusters = 5  # Set the number of clusters to find
kmeans_serial = KMeans(n_clusters=n_clusters, random_state=42)  # random_state ensures reproducibility

# Measure the time taken to train the KMeans model (serial execution)
start_time = time.time()  # Start the timer
kmeans_serial.fit(X_serial)  # Train the model
train_time = time.time() - start_time  # Calculate the elapsed time

# Measure the time taken for prediction (serial execution)
start_time = time.time()  # Reset the timer
y_pred_serial = kmeans_serial.predict(X_serial)  # Predict cluster labels for all data points
predict_time = time.time() - start_time  # Calculate the elapsed time for prediction

# Calculate the model's inertia (sum of squared distances of samples to their closest cluster center)
inertia_serial = kmeans_serial.inertia_

# Output the results of training and prediction
print(f"Training time (serial): {train_time:.2f} seconds")  # Display how long it took to train the model
print(f"Prediction time (serial): {predict_time:.2f} seconds")  # Display how long it took to make predictions
print(f"Inertia (serial): {inertia_serial:.2f}")  # Display the model's inertia, indicating clustering performance
