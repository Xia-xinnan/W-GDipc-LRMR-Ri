from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from data_helper import x, y
import warnings

warnings.filterwarnings('ignore')


def find_best_n_features_ridge(n=8, out_path=''):

    ridge = LogisticRegression(penalty='l2', class_weight='balanced')

    params_tuned = {'C': [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 180, 190,
                          200, 220, 240, 250, 260, 300, 500, 1000, 2000]}

    clf = GridSearchCV(ridge, params_tuned, scoring='accuracy', cv=5)
    clf.fit(x, y)
    print('During Ridge, the penalty parameter alpha is set as', clf.best_params_['C'])
    clf.best_estimator_.fit(x, y)
    clf.best_estimator_.coef_ = np.abs(clf.best_estimator_.coef_)

    best_n = []
    # best_n_coef = []
    for i in range(n):
        best_position = np.nanargmax(clf.best_estimator_.coef_)
        best_n.append(best_position)
        # best_n_coef.append(copy.deepcopy(clf.best_estimator_.coef_[:, best_position][0]))
        clf.best_estimator_.coef_[:, best_position] = np.nan

    print('Selected Features:', best_n)
    best_features = x[:, best_n]
    print('Shape of features selected:', best_features.shape)
    best_features_with_label = pd.DataFrame(np.concatenate([best_features, y.reshape(len(y), 1)], axis=1))

    out_path = out_path + 'ridge_best_' + str(n) + '.csv'
    best_features_with_label.to_csv(out_path, header=None, index=None)


if __name__ == '__main__':
    find_best_n_features_ridge(8)
