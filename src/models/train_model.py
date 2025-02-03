
# Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

# Import Database
data = pd.read_csv("../data/meteo.csv")

X = data.drop(columns=["RainTomorrow"])
X = X.astype("float")

y = data["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=data['RainTomorrow'])

# Standardize the original training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
params = {"n_estimators": 10, "max_depth": 10, "random_state": 42}
rfc = RandomForestClassifier(**params)
rfc.fit(X_train, y_train)

# Model evaluation
y_pred = rfc.predict(X_test)
y_probs = rfc.predict_proba(X_test_scaled)[:,1]

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_probs)
conf_mat = metrics.confusion_matrix(y_test, y_pred)
pr_auc = metrics.average_precision_score(y_test, y_probs)
metrics_rfc = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1, "ROC AUC": roc_auc, "Confusion Matrix": conf_mat, "PR AUC": pr_auc}

print(metrics_rfc)

with open("../model/rfc.pkl", "wb") as file:
    pickle.dump(rfc, file)

