import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import pickle

df = pd.read_csv("StudentsPerformance.csv")

df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['race/ethnicity'] = df['race/ethnicity'].map({
    'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4
})
df['parental level of education'] = df['parental level of education'].map({
    'some high school': 0, 'high school': 1, 'some college': 2,
    "associate's degree": 3, 'bachelor\'s degree': 4, 'master\'s degree': 5
})
df['lunch'] = df['lunch'].map({'standard': 0, 'free/reduced': 1})
df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})

features = [
    "gender", "race/ethnicity", "parental level of education", 
    "lunch", "test preparation course", "math score", 
    "reading score", "writing score"
]
X = df[features].values

df['total_score'] = df['math score'] + df['reading score'] + df['writing score']

def categorize_score(total_score):
    if total_score < 150:
        return 0  # Fail
    elif 150 <= total_score < 200:
        return 1  # Average
    elif 200 <= total_score < 250:
        return 2  # Good
    else:
        return 3  # Excellent

df['label'] = df['total_score'].apply(categorize_score)
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:k]
        classes = [label for _, label in nearest_neighbors]
        predicted_class = np.bincount(classes).argmax()
        predictions.append(predicted_class)
    return np.array(predictions)

y_pred = knn_predict(X_train, y_train, X_test, k=3)

model_data = {
    "X_train": X_train,
    "y_train": y_train,
    "k": 3
}
with open("student_knn_multiclass_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

def confusion_matrix(y_true, y_pred, num_classes=4):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix

conf_matrix = confusion_matrix(y_test, y_pred, num_classes=4)

print("Confusion Matrix:")
print(conf_matrix)

class_names = ["Fail", "Average", "Good", "Excellent"]
for i in range(conf_matrix.shape[0]):
    TP = conf_matrix[i][i]
    FN = np.sum(conf_matrix[i, :]) - TP
    FP = np.sum(conf_matrix[:, i]) - TP
    TN = np.sum(conf_matrix) - (TP + FN + FP)
    
    class_precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    class_recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    class_f1_score = (
        2 * (class_precision * class_recall) / (class_precision + class_recall)
        if (class_precision + class_recall) != 0
        else 0
    )

    print(f"\nClass {i} ({class_names[i]}):")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Precision: {class_precision:.2f}")
    print(f"Recall: {class_recall:.2f}")
    print(f"F1 Score: {class_f1_score:.2f}")

mcc = matthews_corrcoef(y_test, y_pred)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.2f}")

def load_model_and_predict(new_data_point):
    with open("student_knn_multiclass_model.pkl", "rb") as file:
        model_data = pickle.load(file)
    
    X_train = model_data["X_train"]
    y_train = model_data["y_train"]
    k = model_data["k"]
    
    prediction = knn_predict(X_train, y_train, np.array([new_data_point]), k=k)
    return prediction[0]

new_data_point = [1, 2, 4, 0, 1, 85, 90, 95] 
new_prediction = load_model_and_predict(new_data_point)
print(f"\nPrediction for new data point {new_data_point}: Class {new_prediction} ({class_names[new_prediction]})")
