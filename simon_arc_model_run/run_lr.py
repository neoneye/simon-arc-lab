"""
Experiments with Logistic Regression

requires scikit-learn
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Generate synthetic data
num_samples = 1000      # Number of samples
num_features = 200      # Number of input parameters (features)
num_classes = 10        # Number of classes

# Randomly generate input features
X = np.random.rand(num_samples, num_features)

# Randomly generate labels from 0 to 9
y = np.random.randint(0, num_classes, num_samples)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
