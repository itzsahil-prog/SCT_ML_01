import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("train.csv")

# Select only important features (simple beginner version)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual']
target = 'SalePrice'

data = data[features + [target]].dropna()

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Check accuracy
score = model.score(X_test, y_test)
print("Model Accuracy (R² Score):", score)

# Save model
pickle.dump(model, open("house_model.pkl", "wb"))

print("Model saved successfully!")
