#!/usr/lib/evn python3
# -*- coding: utf-8 -*-

from math import log
from sklearn import datasets


def calculate_ent(data_set):
    """
    计算样本集的信息熵
    :param data_set: 样本集
    :return: 信息熵
    """
    data_set_length = len(data_set)
    class_list = [example[-1] for example in data_set]   # 结果分类列表
    class_count = {}                                     # 每个结论类的数量
    # 遍历所有的样本, 计算每种分类的数量
    for key in class_list:
        if key not in class_count.keys():
            class_count[key] = 0
        class_count[key] += 1
    ent = 0.0
    # 计算信息熵
    for key in class_count:
        prob = float(class_count[key]) / data_set_length
        ent -= prob * log(prob, 2)
    return ent


def split_data_set(data_set, axis, value):
    """
    通过属性和属性值提取样本
    :param data_set: 样本集
    :param axis: 属性
    :param value: 属性值
    :return: 提取的样本集
    """
    return_data_set = []
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            # 提取去掉属性axis后的样本
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            return_data_set.append(reduced_feature_vector)
    return return_data_set


def calculate_gain(data_set, axis):
    """
    计算属性axis的信息增益
    :param data_set:
    :param axis: 属性下标
    :return:
    """
    data_set_ent = calculate_ent(data_set)
    feature_class = [example[axis] for example in data_set]
    unique_feature_class = set(feature_class)
    current_ent = data_set_ent
    data_set_length = len(data_set)
    for feature in unique_feature_class:
        feature_data_set = split_data_set(data_set, axis, feature)
        feature_data_set_ent = calculate_ent(feature_data_set)
        current_ent -= float(len(feature_data_set)) / data_set_length * feature_data_set_ent
    return current_ent


def calculate_intrinsic_value(data_set, axis):
    """
    计算属性axis的固有值
    :param data_set:
    :param axis: 属性下标
    :return:
    """
    feature_class = [example[axis] for example in data_set]
    unique_feature_class = set(feature_class)
    data_set_length = float(len(data_set))
    iv = 0.0
    for feature in unique_feature_class:
        feature_data_set = split_data_set(data_set, axis, feature)
        feature_data_set_length = len(feature_data_set)
        iv -= feature_data_set_length / data_set_length * log(feature_data_set_length / data_set_length, 2)
    return iv


def calculate_gain_ratio(data_set, axis):
    """
    计算属性axis信息增益率
    :param data_set:
    :param axis: 属性下标
    :return:
    """
    gain = calculate_gain(data_set, axis)
    iv = calculate_intrinsic_value(data_set, axis)
    return gain / iv


def choose_max_gain_ratio(data_set):
    """
    选取"增益增益率"gain_ratio最大的属性
    :param data_set: 样本集
    :return: 属性
    """
    feature_length = len(data_set[0]) - 1
    max_feature_gain_ratio = 0.0     # 初始最大信息增益
    max_feature_axis = -1      # 初始的产生最大信息增益的属性下标
    # 计算每个属性的信息增益s
    for i in range(feature_length):
        # 计算对于属性i的信息增益
        current_gain_ratio = calculate_gain(data_set, i)
        if max_feature_gain_ratio < current_gain_ratio:
            max_feature_gain_ratio = current_gain_ratio
            max_feature_axis = i
    return max_feature_axis


def majority_count(data_set):
    """
    计算样本集中数量最多的分类
    :param data_set: 样本集
    :return:
    """
    class_count = {}
    for vector in data_set:
        clazz = vector[-1]
        if clazz not in class_count.keys():
            class_count[clazz] = 0
        class_count[clazz] += 1
    max_count = 0
    max_class = data_set[0][-1]
    for clazz, count in class_count:
        if count > max_count:
            max_class = clazz
            max_count = count
    return max_class


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    # 当数据集中类别完全一样时直接返回
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 属性为空时返回数量最多的类
    if len(data_set[0]) == 1:
        return majority_count(data_set)
    # 获取最优属性
    axis = choose_max_gain_ratio(data_set)
    axis_feature_list = [example[axis] for example in data_set]
    label = labels[axis]
    del labels[axis]
    node = {label: {}}
    axis_unique_feature_list = set(axis_feature_list)
    # 对每一个属性作判断
    for feature in axis_unique_feature_list:
        feature_data_set = split_data_set(data_set, axis, feature)
        # 属性上无分类时选取数量最多的类作为分类
        if len(feature_data_set) == 0:
            majority_class = majority_count(feature_data_set)
            node[label][feature] = majority_class
        else:
            node[label][feature] = create_tree(feature_data_set, labels)
    labels.insert(axis, label)
    return node


def create_data_set():
    file = open('/home/mejhwu/workspace/machine_learning/breast-cancer-wisconsin.data', 'r')
    data = []
    line = file.readline()
    while line:
        line = line[:-1]
        line_data = line.split(',')
        line_data.pop(0)
        data.append(line_data)
        line = file.readline()
    # labels = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
    #           'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    # label为属性的下标
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    return data, labels


def test_data_vec(tree, data_vec):
    # 当tree不是对象时,说明已经判断到叶子节点, 可直接判断预测结果是否正确
    if not isinstance(tree, dict):
        if tree == data_vec[-1]:
            return 1
        else:
            return 0
    # 构建的树以dict的形式存储, keys只有一个
    # 获取dict的key, key为属性下表
    key = 0
    for i in tree.keys():
        key = i
    return test_data_vec(tree.get(key).get(data_vec[key]), data_vec)


def test_tree(tree, data_set):
    right_count = 0
    for data_vec in data_set:
        right_count += test_data_vec(tree, data_vec)
    return float(right_count) / len(data_set)


if __name__ == '__main__':
    all_data_set, all_labels = create_data_set()
    train_data_set = all_data_set[:400]
    test_data_set = all_data_set[400:]
    gian_tree = create_tree(train_data_set, all_labels)
    right_rate = test_tree(gian_tree, test_data_set)
    print(right_rate)