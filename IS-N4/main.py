import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# 1. Branje podatkov
df = pd.read_csv("podatki.csv", header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 2. Kodiranje oznak
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Delitev na učni in testni del
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Seznam modelov
modeli = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Neural Net": MLPClassifier(max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# 5. Testiranje
print("Rezultati klasifikatorjev:\n")
for ime, model in modeli.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{ime}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")



if __name__ == '__main__':
    pass
