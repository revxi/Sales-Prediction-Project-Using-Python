# sales_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('dataset/advertising.csv')  # Ensure the CSV is placed in 'dataset/' folder

# Explore data
print("\nData Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Optional: Visualize correlations
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Define feature and target
X = df[['TV']]  # Try with 'Radio' or 'Newspaper' as well
y = df['Sales']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Visualize prediction
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Sales')
plt.title("Actual vs Predicted Sales")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.legend()
plt.savefig("images/output_plot.png")  # Save plot to images folder
plt.show()
