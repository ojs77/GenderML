import pandas as pd

df = pd.read_csv('data.csv')

# Removes outliers and any other unwanted examples
def clean_data(data):
    rows_1 = []

    for row in reversed(range(len(data))):
        if -1 in df.iloc[row].values:
            rows_1.append(row)        
        elif 0 in df.iloc[row].values:
            rows_1.append(row)        
        elif df.iloc[row]["age"] > 100:
            rows_1.append(row)
        elif df.iloc[row]["gender"] == 3:
            rows_1.append(row)
    data = data.drop(labels = rows_1, axis = 0)
    return data

# Splits data into independent and dependent variables
# Used the 4 attriubutes as they are dependent upon Q1-32, is this feature selection
# Using just the final 3 attributes currently as they seem more correlated with gender
def extract_data(data, category):
    X = data.iloc[:, 33: -3]
    y = data[category]
    return X, y

df = clean_data(df)
X, y = extract_data(df, "gender")

# # # Splits data into train, val and test sets
def double_split(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state= 2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.8, random_state= 2, stratify= y_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = double_split(X, y)

# Imbalanced Data: Age is spread out and gender is 55/45 split between the 2
# Stratified T-T Split: Working first with age which is numeric, will apply this to working with gender


# Transform raw data to feature vector: 
# Feature Vector is formatted: X: [[Affiliative, Selfenhancing, agressive, selfdefeating], ..., ...], y: [Gender, ..., ...]
import numpy as np
X_train, X_val, X_test, y_train, y_val, y_test = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline using KNN:
def knn_pipeline(X_train, y_train, X_val, y_val): 
    from sklearn.neighbors import KNeighborsClassifier
    clf_KNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))

    clf_KNN.fit(X_train, y_train)

    def k_knn(X_train, y_train):
        # Calculating error for K values between 1 and 40:
        err = []
        for i in range(1, 40):
            clf_KNN = KNeighborsClassifier(n_neighbors=i)
            clf_KNN.fit(X_train, y_train)
            pred_i = clf_KNN.predict(X_val)
            err.append(np.mean(pred_i != y_val))

        # Plotting variation of error against K:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, 40), err, color='blue', marker='o',
                markerfacecolor='black', markersize=10)
        plt.title('Error Rate vs K Value', fontweight = "bold")
        plt.xlabel('K Value', fontweight = "bold")
        plt.ylabel('Mean Error', fontweight = "bold")
        plt.show()

    # k_knn(X_train, y_train)

    # Analysis:
    # This shows 16 as the best value for K, as it minimises the error, 
    # but it does this by applying more bias and as such predicting males more and females less.
    #25 has a higher accuracy and much better at predicting females

    return clf_KNN

# Pipeline using Linear SVC:
def lsvc_pipeline(X_train, y_train):
    from sklearn.svm import LinearSVC

    clf_lsvc = make_pipeline(StandardScaler(), LinearSVC(random_state=0, class_weight="balanced"))

    clf_lsvc.fit(X_train, y_train)

    return clf_lsvc

# Pipeline using RBF SVC:
def rbf_pipeline(X_train, y_train):
    from sklearn.svm import SVC

    clf_rbf = make_pipeline(StandardScaler(), SVC(random_state=0, class_weight="balanced"))

    clf_rbf.fit(X_train, y_train)

    return clf_rbf

# Pipeline using RBF SVC Ensemble Model:
def ensemble_pipeline(X_train, y_train):
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier


    clf_bc = BaggingClassifier(base_estimator=SVC(class_weight="balanced", random_state=0), n_estimators=10, random_state=0)

    clf_pipeline = make_pipeline(StandardScaler(), clf_bc)

    clf_pipeline.fit(X_train, y_train)

    return clf_pipeline

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Evaluating Models
def classifier_eval(clf, X_val, y_val):
     # Testing the algorithm by optaining the score and accuracy for each class + overall
    # print(f"{str(clf.steps[1][0])} Score: {round(clf.score(X_val, y_val), 4)}")
    count_correct_1 = 0
    count_false_1 = 0
    count_correct_2 = 0
    count_false_2 = 0
    
    for z in range(len(X_val)):
        if y_val[z] == 1:
            if clf.predict([X_val[z]]) == y_val[z]:
                count_correct_1 += 1
            else:
                count_false_1 += 1
        elif y_val[z] == 2:
            if clf.predict([X_val[z]]) == y_val[z]:
                count_correct_2 += 1
            else:
                count_false_2 += 1

    print(f"""Male: Correct: {count_correct_1}, False: {count_false_1}, (%): {round(count_correct_1/(count_correct_1+count_false_1)*100, 2)}%,
    Female: Correct: {count_correct_2}, False: {count_false_2}, (%): {round(count_correct_2/(count_correct_2+count_false_2)*100, 2)}%, 
    Total Correct: {round((count_correct_1+count_correct_2)/(count_correct_1+count_correct_2+count_false_1 + count_false_2)*100, 2)}%""")

    # if str(clf.steps[1][0]) == "linearsvc":
        # print(clf.named_steps['linearsvc'].coef_)
        # print(clf.named_steps['linearsvc'].intercept_)


    y_pred = clf.predict(X_val)

    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    # print(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))


def cv_scores():
    from sklearn.model_selection import cross_val_score
    cv_scores_lsvc = cross_val_score(lsvc_pipeline(X_train, y_train), X_train, y_train, cv=10)
    cv_scores_rbf = cross_val_score(rbf_pipeline(X_train, y_train), X_train, y_train, cv=10)
    cv_scores_knn = cross_val_score(knn_pipeline(X_train, y_train, X_val, y_val), X_train, y_train, cv=10)
    cv_scores_ensemble = cross_val_score(ensemble_pipeline(X_train, y_train), X_train, y_train, cv=10)
    

    return f"CV Score: LSVC: {round(cv_scores_lsvc.mean(), 4)}, RBF: {round(cv_scores_rbf.mean(), 4)}, KNN: {round(cv_scores_knn.mean(), 4)} Ensemble: {round(cv_scores_ensemble.mean(), 4)}"


classifier_eval(lsvc_pipeline(X_train, y_train), X_test, y_test)
classifier_eval(rbf_pipeline(X_train, y_train), X_val, y_val)
classifier_eval(knn_pipeline(X_train, y_train, X_val, y_val), X_test, y_test)
classifier_eval(ensemble_pipeline(X_train, y_train), X_test, y_test)
print(cv_scores())