import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
mnist = pd.read_csv('mnist.csv/mnist.csv')
X, y = mnist.iloc[:, 1:], mnist.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Display results
print(f"Accuracy: {accuracy_score(y_test, pred)*100:.2f}%\n")
for digit in range(10):
    recall = classification_report(y_test, pred, output_dict=True)[str(digit)]['recall']
    print(f"Digit {digit}: {recall*100:.1f}% correctly identified.")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Show image
if input("\nShow a test image? (yes/no): ").strip().lower() == 'yes':
    idx = int(input(f"Enter index (0-{len(X_test)-1}): "))
    img = X_test.iloc[idx].values.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y_test.iloc[idx]}, Predicted: {pred[idx]}")
    plt.axis('off')
    plt.show()
