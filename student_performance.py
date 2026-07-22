import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_performance.csv")

# Input (Study Hours) and Output (Marks)
X = data[["StudyHours"]]
y = data["Marks"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 7.5 study hours
hours = [[7.5]]
prediction = model.predict(hours)

print("Predicted Marks:", prediction[0])

# Plot the graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Performance Prediction")
plt.legend()
plt.show()