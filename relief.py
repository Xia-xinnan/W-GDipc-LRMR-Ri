import numpy as np
import pandas as pd
from data_helper import x, y
import warnings
import copy
from sklearn.preprocessing import scale
from scipy.spatial.distance import euclidean

warnings.filterwarnings('ignore')


def find_best_n_features_relief(n=8, out_path=''):

    relief_all = calculate_relief(x, y)
    
    # 找出8个最大的
    best_n = []
    best_n_ref = []
    for i in range(n):
        best_position = np.nanargmax(relief_all)
        best_n.append(best_position)
        best_n_ref.append(copy.deepcopy(relief_all[best_position]))
        relief_all[best_position] = np.nan

    print('Found', n, 'features with largest Relief Statistics, whose positions are:')
    print(best_n)
    print()
    print('The Relief Statistics of these features are:')
    print(best_n_ref)
    print()
    
    best_features = x[:, best_n]
    print('Shape of features selected:', best_features.shape)
    best_features_with_label = pd.DataFrame(np.concatenate([best_features, y.reshape(len(y), 1)], axis=1))

    out_path = out_path + 'relief_best_' + str(n) + '.csv'
    best_features_with_label.to_csv(out_path, header=None, index=None)


def calculate_relief(x, y):
    # 特征归一化
    x_scaled = scale(x)
    # np.std(x_scaled, axis=0)

    # 遍历每个sample，寻找hit和miss，用欧氏距离
    dis_hit = np.ones((len(x_scaled), len(x_scaled))) * np.inf
    dis_miss = np.ones((len(x_scaled), len(x_scaled))) * np.inf
    for i in range(len(x_scaled)):
        for j in range(len(x_scaled)):
            if i == j:
                continue
            else:
                if y[i] == y[j]:
                    dis_hit[i, j] = euclidean(x_scaled[i, :], x_scaled[j, :])
                else:
                    dis_miss[i, j] = euclidean(x_scaled[i, :], x_scaled[j, :])
    
    x_hit = np.argmin(dis_hit, axis=1)
    # print([_ for _ in range(len(x_hit)) if x_hit[_] == _])
    x_miss = np.argmin(dis_miss, axis=1)
    # print([_ for _ in range(len(x_miss)) if x_miss[_] == _])
    
    # 计算统计量
    relief_ij = np.zeros(x_scaled.shape)
    for i in range(len(x_scaled)):
        dis_xi_to_hit = (x_scaled[i, :] - x_scaled[x_hit[i], :]) ** 2
        dis_xi_to_miss = (x_scaled[i, :] - x_scaled[x_miss[i], :]) ** 2
        relief_ij[i, :] = dis_xi_to_miss - dis_xi_to_hit
    
    relief_all = np.sum(relief_ij, axis=0)
    # print(relief_all.shape)
    return relief_all


if __name__ == '__main__':
    find_best_n_features_relief(8)
