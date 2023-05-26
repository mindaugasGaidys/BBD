import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

test_data = pd.read_excel("Data/Datasets/NB.xlsx", sheet_name="NB_test")

test_data = test_data.dropna(subset=['text_cleaned', 'category'])

X_test = test_data['text_cleaned']
y_test = test_data['category']

vectorizer, selector, best_model = joblib.load('NB/parametersData/best_NB.joblib')

X_test_vec = vectorizer.transform(X_test)
X_test_vec_selected = selector.transform(X_test_vec)
y_pred = best_model.predict(X_test_vec_selected)
y_proba = best_model.predict_proba(X_test_vec_selected)

with open("NB/parametersData/Nb_results_best.txt", "r") as f:
    lines = f.readlines()
    best_score = float(lines[1].split(':')[1].strip())

report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

with open("NB/FinalResults/NB_test_info_recall.txt", "w") as f:
    f.write(f"Best Alpha: {best_model.alpha}\n")
    f.write(f"Best Recall for Category 2: {best_score}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(matrix))
    f.write(f"\nAccuracy Score: {accuracy}")

with open("NB/FinalResults/NB_class_prob.txt", "w") as f:
    np.savetxt(f, y_proba)
