import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import time

# Define a function to run KMeans clustering on a chunk of data
def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize KMeans model
    start_time = time.time()  # Start timer for training
    kmeans.fit(data)  # Fit the model on the data chunk
    train_time = time.time() - start_time  # Calculate the training time
    inertia = kmeans.inertia_  # Get the inertia for this chunk
    return kmeans.labels_, kmeans.cluster_centers_, train_time, inertia  # Return labels, centers, training time, and inertia

# Load the dataset
data = pd.read_csv('synthetic_large_clustering_data.csv')

# Select two features for clustering
X_trainer = data[['Feature_1', 'Feature_2']].values  # Extract Feature_1 and Feature_2 for training

# Set up parallel processing
n_jobs = 2  # Define the number of parallel jobs (processors) to use
data_chunks = np.array_split(X_trainer, n_jobs)  # Split the data into chunks for parallel processing

# Run KMeans on each data chunk in parallel and measure training time
results = Parallel(n_jobs=n_jobs)(delayed(run_kmeans)(chunk, n_clusters=2) for chunk in data_chunks)

# Measure prediction time separately (predicting on the entire dataset)
start_pred_time = time.time()
all_labels = np.concatenate([labels for labels, _, _, _ in results])  # Concatenate labels from each chunk
prediction_time = time.time() - start_pred_time  # Calculate prediction time

# Add the cluster labels to the original dataframe
data['cluster'] = all_labels

# Calculate the total training time by summing up individual chunk times
training_time = sum(train_time for _, _, train_time, _ in results)

# Calculate the total inertia by summing up individual chunk inertias
total_inertia = sum(inertia for _, _, _, inertia in results)

# Output the execution times and inertia
print(f"\nTraining Time (Parallel): {training_time:.4f} seconds")  # Display the total training time
print(f"Prediction Time (Parallel): {prediction_time:.4f} seconds")  # Display the prediction time
print(f"Total Inertia (Parallel): {total_inertia:.4f}")  # Display the total inertia
