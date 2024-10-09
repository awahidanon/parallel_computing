import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Load the CSV file
df = pd.read_csv('synthetic_classification_data.csv')

# Split the DataFrame into features (X) and labels (y)
X = df.drop("Label", axis=1)  # All columns except 'Label' are features
y = df["Label"]               # The 'Label' column is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model with n_jobs=-1 to use all cores
model_multi_core = RandomForestClassifier(n_jobs=-1)

# Time the training process
start_time = time.time()
model_multi_core.fit(X_train, y_train)
train_time = time.time() - start_time

# Time the prediction process
start_time = time.time()
y_pred = model_multi_core.predict(X_test)
predict_time = time.time() - start_time

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision 
precision = precision_score(y_test, y_pred, average='macro')  

# Output the results
print(f"Training time (all cores): {train_time:.2f} seconds")
print(f"Prediction time (all cores): {predict_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
