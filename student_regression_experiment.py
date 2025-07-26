import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt # Added for plotting

# Student Exam Scores Prediction

# Data setup
np.random.seed(42) 

n_samples = 100
study_hours = np.random.uniform(10, 100, n_samples)

# True relationship: Exam_Score = 25 + 0.7 * Study_Hours + noise
true_b0 = 25
true_b1 = 0.7
noise = np.random.normal(0, 8, n_samples) 

exam_scores = true_b0 + true_b1 * study_hours + noise

X = study_hours.reshape(-1, 1)
y = exam_scores

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Results
print("--- Student Exam Scores Prediction ---")
print("Model coefficients:")
print("  Intercept:", round(model.intercept_, 2))
print("  Study Hours Coef:", round(model.coef_[0], 2))

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
print("R-squared on test set:", round(r2, 2))

# Quick prediction example
example_hours = np.array([[70]])
predicted_score = model.predict(example_hours)
print("Predicted score for 70 hours of study:", round(predicted_score[0], 2))

# My thoughts on this:
print("\nInterpretation:")
print("Looks like more study hours really do help with exam scores. The model picked up on that.")
print("For each extra hour, scores go up by about", round(model.coef_[0], 2), "points. The R-squared of", round(r2, 2), "is pretty good, means the model explains most of the score variation.")
print("It confirms what you'd expect, really.")

# --- Plotting the results ---
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.6, label='Generated Data')
plt.plot(X, model.predict(X), color='red', label='Fitted Regression Line')
plt.title('Student Exam Scores vs. Study Hours')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()


