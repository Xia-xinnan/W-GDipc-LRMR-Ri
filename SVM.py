from svmutil import *
from grid import *
from numpy import *


rate, param = find_parameters('adpaTr2.txt')
rate, param = find_parameters('adpaTr2.txt', '-log2c -5,5,0.1 -log2g -5,5,0.1')
#y, x = svm_read_problem('adpaTr2.txt')
#m = svm_train(y, x ,'-c 5 -g 0.002 ')
#yt,xt=svm_read_problem('adpaTe2.txt')
#p_label, p_acc, p_val = svm_predict(yt, xt, m)