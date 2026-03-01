import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# User input for train/test split
train_percent = int(input("Enter train percentage (e.g., 80 for 80%): "))
test_percent = 100 - train_percent
print(f"Using {train_percent}% train and {test_percent}% test split.")

# Load wine quality dataset (red wine)
wine = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
X = wine.drop('quality', axis=1)
y = wine['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percent/100, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter grids
lr_params = {'C': [0.1, 1, 10], 'solver': ['lbfgs']}
dt_params = {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]}
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
svm_params = {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}
knn_params = {'n_neighbors': [3, 5, 7, 9]}

# GridSearchCV for each model
print("Tuning models with GridSearchCV...")
gs_lr = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=3)
gs_lr.fit(X_train_scaled, y_train)
best_lr = gs_lr.best_estimator_
lr_preds = best_lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"Tuned Logistic Regression Accuracy: {lr_acc*100:.2f}% (Best params: {gs_lr.best_params_})")

# Decision Tree
gs_dt = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=3)
gs_dt.fit(X_train, y_train)
best_dt = gs_dt.best_estimator_
dt_preds = best_dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
print(f"Tuned Decision Tree Accuracy: {dt_acc*100:.2f}% (Best params: {gs_dt.best_params_})")

# Random Forest
gs_rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=3)
gs_rf.fit(X_train, y_train)
best_rf = gs_rf.best_estimator_
rf_preds = best_rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Tuned Random Forest Accuracy: {rf_acc*100:.2f}% (Best params: {gs_rf.best_params_})")

# SVM
gs_svm = GridSearchCV(SVC(), svm_params, cv=3)
gs_svm.fit(X_train_scaled, y_train)
best_svm = gs_svm.best_estimator_
svm_preds = best_svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"Tuned SVM Accuracy: {svm_acc*100:.2f}% (Best params: {gs_svm.best_params_})")

# KNN
gs_knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3)
gs_knn.fit(X_train_scaled, y_train)
best_knn = gs_knn.best_estimator_
knn_preds = best_knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_preds)
print(f"Tuned KNN Accuracy: {knn_acc*100:.2f}% (Best params: {gs_knn.best_params_})")

# Visualizations (for tuned models)
def plot_scatter(X_vis, y_vis, name, scaled=False):
    plt.figure(figsize=(6,4))
    if isinstance(X_vis, pd.DataFrame):
        plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis, cmap='viridis', alpha=0.7)
        plt.xlabel(('Scaled ' if scaled else '') + X_vis.columns[0])
        plt.ylabel(('Scaled ' if scaled else '') + X_vis.columns[1])
    else:
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='viridis', alpha=0.7)
        plt.xlabel(('Scaled ' if scaled else '') + X_train.columns[0])
        plt.ylabel(('Scaled ' if scaled else '') + X_train.columns[1])
    plt.title(f'Training Data Visualization for {name} (Tuned)')
    plt.colorbar(label='Quality')
    plt.show()

plot_scatter(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train, "Logistic Regression", scaled=True)
plot_scatter(X_train, y_train, "Decision Tree")
plot_scatter(X_train, y_train, "Random Forest")
plot_scatter(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train, "SVM", scaled=True)
plot_scatter(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train, "KNN", scaled=True)
