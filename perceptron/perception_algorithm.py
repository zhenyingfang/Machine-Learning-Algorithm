'''
@Author: fzy
@Date: 2019-05-23 14:53:54
@LastEditors: Zhenying
@LastEditTime: 2019-05-23 16:51:01
@Description: 
'''
import numpy as np
import pandas as pd
import time
import argparse
from utils.log_utils import get_logger
from utils.color_utils import to_blue, to_cyan, to_green, to_magenta, to_red, to_yellow

# ===== 加载数据集 =====
def load_data(filename, logg):
    # logg.info(to_blue("===== Loading Data ====="))
    logg.info("===== Loading Data =====")
    df = pd.read_csv(filename, header=None)
    # 获得类别标签
    labels = df.iloc[:, 0].values
    # 获得数据
    datas = df.iloc[:, 1:].values
    # 转换成二分类，分0类和非0类，将原始类别为0的标记为1，原始类别非0的标记为-1
    labels = np.where(labels > 0, 1, -1)
    # 将数据除255
    datas = datas / 255.
    # logg.info(to_blue("===== Loaded Data  ====="))
    logg.info("===== Loaded Data  =====")
    return datas, labels

# ===== 感知机训练算法 =====
def perceptron(datas, labels, logg, args):
    # logg.info(to_cyan("===== start train ====="))
    logg.info("===== start train =====")
    # 得到训练数据的数量和维度
    m, n = datas.shape
    # 初始化权重和偏置
    w = np.zeros((1, n))
    b = 0
    # 初始化学习率
    eta = args.eta
    # 进行iter次迭代计算
    for now_iter in range(args.iters):
        err_cnt = 0.
        for i in range(m):
            xi = datas[i]
            yi = labels[i]
            xi = np.mat(xi)
            yi = np.mat(yi)
            # 判断是否是误分类样本
            if (-1 * yi * (w * xi.T + b)) >= 0:
                err_cnt = err_cnt + 1
                # 对于误分类样本，进行梯度下降，更新w和b
                w = w + eta * yi * xi
                b = b + eta * yi
        logg.info('Iter [{0}]:[{1}] Train Err: [{2:.4f}]'.format(now_iter, args.iters, err_cnt / float(m)))
        # logg.info(to_green('Iter [{0}]:[{1}] Train Err: [{2}]'.format(to_yellow(now_iter),
        #                                                               to_cyan(args.iters),
        #                                                               to_blue('{0:.4f}'.format(err_cnt / float(m))))))
    # logg.info(to_cyan("===== trained ====="))
    logg.info("===== trained =====")
    return w, b

# ===== 测试代码 =====
def val(datas, labels, w, b, logg):
    # logg.info(to_magenta("===== start testing ====="))
    logg.info("===== start testing =====")
    m, n = datas.shape
    # 用来统计预测错误的个数
    errorCnt = 0
    # 对所有样本进行预测
    for i in range(m):
        xi = datas[i]
        yi = labels[i]
        xi = np.mat(xi)
        yi = np.mat(yi)
        res = -1 * yi * (w * xi.T + b)
        if res >= 0:
            errorCnt += 1
    accruRate = 1 - (errorCnt / m)
    # logg.info(to_magenta("===== tested ====="))
    # logg.info(to_red("accRate: {0}".format(accruRate)))
    logg.info("===== tested =====")
    logg.info("accRate: {0}".format(accruRate))
    return accruRate


if __name__ == "__main__":
    # ===== 初始化参数 =====
    parser = argparse.ArgumentParser(description="perceptron")
    parser.add_argument("--eta", default=0.0001, type=int,
                        help="eta")
    parser.add_argument("--iters", default=100, type=int,
                        help="iters")
    args = parser.parse_args()
    # ===== 获取logger =====
    logger = get_logger("perceptron")
    # ===== 读取训练数据 =====
    train_datas, train_labels = load_data("../data/mnist_train.csv", logger)
    # ===== 训练，并返回训练好的权重和偏置 =====
    w, b = perceptron(train_datas, train_labels, logger, args)
    # ===== 读取测试数据 =====
    test_datas, test_labels = load_data("../data/mnist_test.csv", logger)
    # ===== 得到测试结果 =====
    accRate = val(test_datas, test_labels, w, b, logger)
