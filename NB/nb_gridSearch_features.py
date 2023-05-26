import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import make_scorer, recall_score
import numpy as np
import joblib

def recall_category_2(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[2], average='macro')

train_data = pd.read_excel("Data/Datasets/NB.xlsx", sheet_name="Sheet2")

train_data = train_data.dropna(subset=['text_cleaned', 'category'])

X_train = train_data['text_cleaned']
y_train = train_data['category']

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

k = 1500
selector = SelectKBest(chi2, k=k)
selector.fit(X_train_vec, y_train)
X_train_vec_selected = selector.transform(X_train_vec)

param_grid = {
    'alpha': np.logspace(start=-2, stop=1.176, num=1180, base=10)
}

nb_classifier = MultinomialNB()
custom_scorer = make_scorer(recall_category_2)
grid_search = GridSearchCV(nb_classifier, param_grid=param_grid, cv=2, n_jobs=-1, scoring=custom_scorer)

grid_search.fit(X_train_vec_selected, y_train)

best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

joblib.dump((vectorizer, selector, best_model), 'NB/parametersData/best_NB.joblib')

with open("NB/parametersData/NB_grid_results_recall.txt", "w") as f:
    for i, params in enumerate(grid_search.cv_results_['params']):
        alpha = params['alpha']
        mean_test_score = grid_search.cv_results_['mean_test_score'][i]
        std_test_score = grid_search.cv_results_['std_test_score'][i]
        f.write(f"Alpha: {alpha}, Mean Test Score: {mean_test_score}, Std Test Score: {std_test_score}\n")

with open("NB/parametersData/Nb_results_best_recall.txt", "w") as f:
    f.write(f"Best Alpha: {best_model.alpha}\n")
    f.write(f"Best Recall for Category 2: {best_score}\n")
