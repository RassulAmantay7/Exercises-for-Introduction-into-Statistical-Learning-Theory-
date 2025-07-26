import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt # Added for plotting

# Customer Churn Prediction

# Data generation
np.random.seed(42) 

n_customers = 200

# Support calls (0-10)
calls = np.random.randint(0, 11, n_customers)

# Daily usage time (10-200 mins)
usage_time = np.random.uniform(10, 200, n_customers)

# Churn probability based on calls and usage
# Higher calls, lower usage -> higher churn chance
churn_prob = 1 / (1 + np.exp(-(-1 + 0.5 * calls - 0.02 * usage_time)))

# Actual churn (0 or 1)
churn_status = (np.random.rand(n_customers) < churn_prob).astype(int)

data = pd.DataFrame({
    'Calls': calls,
    'UsageTime': usage_time,
    'Churn': churn_status
})

X = data[['Calls', 'UsageTime']]
y = data['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Results
print("--- Customer Churn Prediction ---")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", round(acc, 2))
print("\nMore detailed report:")
print(classification_report(y_test, y_pred))

print("\nModel insights:")
print("  Intercept:", round(model.intercept_[0], 2))
print("  Calls coefficient:", round(model.coef_[0][0], 2))
print("  Usage Time coefficient:", round(model.coef_[0][1], 2))

# My thoughts on this:
print("\nInterpretation:")
print("This model seems to be pretty good at figuring out who's gonna leave. The coefficients make sense:")
print("  - More calls to support (positive coef) means higher chance of leaving.")
print("  - More usage time (negative coef) means lower chance of leaving.")
print("It's cool how logistic regression can pick up on these patterns from just some generated data.")

# --- Plotting the results ---
plt.figure(figsize=(8, 6))

# Plot data points, colored by churn status
plt.scatter(data[data['Churn']==0]['Calls'], data[data['Churn']==0]['UsageTime'], 
            color='blue', label='Stayed (0)', alpha=0.6)
plt.scatter(data[data['Churn']==1]['Calls'], data[data['Churn']==1]['UsageTime'], 
            color='red', label='Churned (1)', alpha=0.6)

# Plot the decision boundary
# The decision boundary is where log_odds = 0, i.e., intercept + coef1*x1 + coef2*x2 = 0
# So, x2 = (-intercept - coef1*x1) / coef2

# Get coefficients
intercept = model.intercept_[0]
coef_calls = model.coef_[0][0]
coef_usage = model.coef_[0][1]

# Create a range for 'Calls' to plot the line
x_calls_range = np.array([data['Calls'].min(), data['Calls'].max()])

# Calculate corresponding 'UsageTime' values for the decision boundary
# Handle case where coef_usage might be very close to zero to avoid division by zero
if abs(coef_usage) > 1e-9:
    x_usage_boundary = (-intercept - coef_calls * x_calls_range) / coef_usage
    plt.plot(x_calls_range, x_usage_boundary, color='green', linestyle='--', 
             label='Decision Boundary')
else:
    # If usage coefficient is negligible, boundary is vertical
    plt.axvline(x=-intercept/coef_calls, color='green', linestyle='--', label='Decision Boundary')

plt.title('Customer Churn Prediction with Logistic Regression')
plt.xlabel('Number of Support Calls')
plt.ylabel('Daily Usage Time (minutes)')
plt.legend()
plt.grid(True)
plt.show()
