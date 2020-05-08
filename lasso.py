from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from data_helper import x, y
import warnings

warnings.filterwarnings('ignore')


def find_best_n_features_lasso(n=8, out_path='', max_iter=100):

    def find_proper_c(n, low, up, max_iter=100):
        c_now = (low + up) / 2
        lasso = LogisticRegression(penalty='l1', class_weight='balanced', C=c_now)
        lasso.fit(x, y)
        # 用二分法缩小范围，直到最后留下刚好n个特征
        count = 0
        fea_num = np.sum(lasso.coef_ > 0.0)
        while count < max_iter and (fea_num > n * 1.05 or fea_num < n):
            if fea_num > n:
                up = (low + up)/2
            elif fea_num < n:
                low = (low + up)/2
            c_now = (low + up)/2
            lasso = LogisticRegression(penalty='l1', class_weight='balanced', C=c_now)
            lasso.fit(x, y)
            fea_num = np.sum(lasso.coef_ > 0.0)
            count += 1
            # print(c_now, ':', fea_num)
        return c_now, count

    low = 0
    up = 100000
    c_now, count = find_proper_c(n, low, up, max_iter)
    retry = 1
    while count == max_iter and retry <= 10:
        print('Didn\'t find proper C within', low, 'and', up, ', retrying', retry)
        low = up
        up = up * 10
        c_now, count = find_proper_c(low, up, max_iter)
        retry += 1

    if retry == 11:
        print('Cound not find proper C, please try a lower dimension.')
        return

    lasso = LogisticRegression(penalty='l1', class_weight='balanced', C=c_now)
    lasso.fit(x, y)
    best_features = x[:, (lasso.coef_ > 0.0)[0]]
    if best_features.shape[1] > n:
        best_n = []
        for i in range(n):
            best_position = np.nanargmax(lasso.coef_)
            best_n.append(best_position)
            lasso.coef_[:, best_position] = np.nan
        best_features = x[:, best_n]

    print('Shape of features selected:', best_features.shape)
    best_features_with_label = pd.DataFrame(np.concatenate([best_features, y.reshape(len(y), 1)], axis=1))

    out_path = out_path + 'lasso_best_' + str(n) + '.csv'
    best_features_with_label.to_csv(out_path, header=None, index=None)


if __name__ == '__main__':
    find_best_n_features_lasso(8)
