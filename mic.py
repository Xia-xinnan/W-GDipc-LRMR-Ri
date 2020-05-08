from minepy import MINE
import numpy as np
import pandas as pd
from data_helper import x, y
import warnings
import copy

warnings.filterwarnings('ignore')


def find_best_n_features_mic(n=8, out_path=''):
    # 计算MIC
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mic_all = []
    for i in range(x.shape[1]):
        xi = x[:, i]
        mine.compute_score(xi, y)
        mic_all.append(mine.mic())

    # 找出8个最大的
    best_n = []
    best_n_mic = []
    for i in range(n):
        best_position = np.nanargmax(mic_all)
        best_n.append(best_position)
        best_n_mic.append(copy.deepcopy(mic_all[best_position]))
        mic_all[best_position] = np.nan

    print('Found', n, 'features with largest MIC, whose positions are:')
    print(best_n)
    print()
    print('The MIC of these features are:')
    print(best_n_mic)
    print()

    best_features = x[:, best_n]
    print('Shape of features selected:', best_features.shape)
    best_features_with_label = pd.DataFrame(np.concatenate([best_features, y.reshape(len(y), 1)], axis=1))

    out_path = out_path + 'mic_best_' + str(n) + '.csv'
    best_features_with_label.to_csv(out_path, header=None, index=None)


if __name__ == '__main__':
    find_best_n_features_mic(8)
