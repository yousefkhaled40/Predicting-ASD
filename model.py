import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def pipeline(x, y, test_size, random_state, shuffle, n_splits, model):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    cnt = 1
    average_accuracy = []
    for train_index, val_index in kf.split(x_train, y_train):
        model = model.fit(x_train[train_index], y_train[train_index])
        y_predicted = model.predict(x_train[val_index])
        acc = accuracy_score(y_train[val_index], y_predicted)
        average_accuracy.append(acc)
        print(f"Fold_{cnt} accuracy: {acc}")
        cnt += 1
        print()
    print('-'*30)
    print(f"The average accuracy for the {n_splits} folds Train/Val: {np.average(average_accuracy)}")
    print('-'*30)
    test_predicted = model.predict(x_test)
    test_acc = accuracy_score(test_predicted, y_test)
    print(f"The final accuracy for the test set: {test_acc}")

