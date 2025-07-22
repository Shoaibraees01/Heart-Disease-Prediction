import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv('data/heart_disease.csv')

# Preprocess data
df.drop_duplicates(inplace=True)
df.fillna(df.median(), inplace=True)

X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Train Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Save models
os.makedirs('app/models', exist_ok=True)
joblib.dump(log_reg, 'app/models/logistic_regression_model.pkl')
joblib.dump(rf_clf, 'app/models/random_forest_model.pkl')
joblib.dump(scaler, 'app/models/scaler.pkl')

# Evaluation
print("\nLogistic Regression:\n", classification_report(y_test, log_reg.predict(X_test)))
print("\nRandom Forest:\n", classification_report(y_test, rf_clf.predict(X_test)))
