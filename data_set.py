import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000000, n_features=20, random_state=42)

# Convert the dataset to a pandas DataFrame
# X contains the features, and y contains the labels
df_X = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
df_y = pd.DataFrame(y, columns=["Label"])

# Concatenate features and labels into a single DataFrame
df = pd.concat([df_X, df_y], axis=1)

# Save the DataFrame to a CSV file
df.to_csv('synthetic_classification_data.csv', index=False)

print("CSV file generated: synthetic_classification_data.csv")
