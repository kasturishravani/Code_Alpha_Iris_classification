import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv("iris.csv")

# Print columns to verify
print("Columns in dataset:")
print(df.columns)

# Remove Id column only if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Target column (last column)
target_col = df.columns[-1]

# Features = all columns except target
X = df.drop(target_col, axis=1)

# Target
y = df[target_col]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy =", round(acc * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Sample Prediction
sample = [X.iloc[0].tolist()]
prediction = model.predict(sample)

print("\nPredicted Species:")
print(le.inverse_transform(prediction)[0])

# Plot
plt.figure(figsize=(6,4))
plt.bar(X.columns, model.feature_importances_)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
