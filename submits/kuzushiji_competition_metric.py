"""
Python equivalent of the Kuzushiji competition metric (https://www.kaggle.com/c/kuzushiji-recognition/)
Kaggle's backend uses a C# implementation of the same metric. This version is
provided for convenience only; in the event of any discrepancies the C# implementation
is the master version.

Tested on Python 3.6 with numpy 1.16.4 and pandas 0.24.2.
Usage: python f1.py --sub_path [submission.csv] --solution_path [groundtruth.csv]
用法：--sub_path D:\mmdetection\submits\biaozhu\det_boxes_dino36.csv --solution_path D:\mmdetection\submits\biaozhu\ground_truth_with_id.csv
"""


import argparse
import multiprocessing
import sys
import csv
import operator
import numpy as np
import pandas as pd

# 这是一个定义命令行解析器的函数，它使用了argparse模块。
def define_console_parser():
    parser = argparse.ArgumentParser() # 创建一个解析器对象
    # 添加两个必须的参数：–sub_path和–solution_path,分别表示提交的预测结果和真实的标注结果的文件路径
    parser.add_argument('--sub_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    return parser

# 定义一个函数，用于计算一个页面的真正例（tp）、假正例（fp）和假负例（fn）
def score_page(preds, truth):
    """
    Scores a single page.
    Args:
        preds: prediction string of labels and center points.
        truth: ground truth string of labels and bounding boxes.
    Returns:
        True/false positive and false negative counts for the page
    """
    # 初始化tp, fp, fn为0
    tp = 0
    fp = 0
    fn = 0

    # 定义一个字典，表示真实的标注结果中每个元素的索引位置，包括label（字符）、X（左上角横坐标）、Y（左上角纵坐标）、Width（宽度）、Height（高度）
    truth_indices = {
        'label': 0,
        'X': 1,
        'Y': 2,
        'Width': 3,
        'Height': 4
    }

    # 定义一个字典，表示提交的预测结果中每个元素的索引位置，包括label（字符）、X（中心点横坐标）、Y（中心点纵坐标）
    preds_indices = {
        'label': 0,
        'X': 1,
        'Y': 2
    }

    # 如果真实的标注结果和提交的预测结果都是空值，即没有任何字符，则返回tp, fp, fn都为0的字典
    if pd.isna(truth) and pd.isna(preds):
        return {'tp': tp, 'fp': fp, 'fn': fn}
    # 如果真实的标注结果是空值，即没有任何字符，则计算提交的预测结果中有多少个字符，即预测结果字符串按空格分割后的长度除以每个元素的个数，并赋值给fp变量
    if pd.isna(truth):
        fp += len(preds.split(' ')) // len(preds_indices)
        return {'tp': tp, 'fp': fp, 'fn': fn} # 返回tp为0，fp为预测字符个数，fn为0的字典
    # 如果提交的预测结果是空值，即没有任何字符
    if pd.isna(preds):
        fn += len(truth.split(' ')) // len(truth_indices) # 计算真实的标注结果中有多少个字符，即标注结果字符串按空格分割后的长度除以每个元素的个数，并赋值给fn变量
        return {'tp': tp, 'fp': fp, 'fn': fn}

    # 将真实的标注结果字符串按空格分割成一个列表
    truth = truth.split(' ')
    if len(truth) % len(truth_indices) != 0:     # 如果列表的长度不能被每个元素的个数整除，即字符串格式不正确
        raise ValueError('Malformed solution string')
    truth_label = np.array(truth[truth_indices['label']::len(truth_indices)])
    truth_xmin = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
    truth_ymin = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
    truth_xmax = truth_xmin + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
    truth_ymax = truth_ymin + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

    preds = preds.split(' ')
    if len(preds) % len(preds_indices) != 0:
        raise ValueError('Malformed prediction string')
    preds_label = np.array(preds[preds_indices['label']::len(preds_indices)])
    preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
    preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)
    preds_unused = np.ones(len(preds_label)).astype(bool)

    for xmin, xmax, ymin, ymax, label in zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label):
        # Matching = point inside box & character same & prediction not already used
        matching = (xmin < preds_x) & (xmax > preds_x) & (ymin < preds_y) & (ymax > preds_y) & (preds_label == label) & preds_unused
        if matching.sum() == 0:
            fn += 1
        else:
            tp += 1
            preds_unused[np.argmax(matching)] = False
    fp += preds_unused.sum()
    return {'tp': tp, 'fp': fp, 'fn': fn}

# 定义一个函数，用于计算F1 score
# def kuzushiji_f1(sub, solution):
#     """
#     Calculates the competition metric.
#     Args:
#         sub: submissions, as a Pandas dataframe
#         solution: solution, as a Pandas dataframe
#     Returns:
#         f1 score
#     """
#     # 如果提交的预测结果和真实的标注结果的数据框中的image_id列的值不完全相同，即图片编号不匹配，则抛出一个值错误异常，提示图片编号不匹配
#     if not all(sub['image_id'].values == solution['image_id'].values):
#         raise ValueError("Submission image id codes don't match solution")
#
#     pool = multiprocessing.Pool()
#     # 使用进程池对象调用score_page函数，传入提交的预测结果和真实的标注结果的数据框中的labels列的值，即预测的标签字符串和真实的标签字符串，返回一个结果列表
#     results = pool.starmap(score_page, zip(sub['labels'].values, solution['labels'].values))
#     pool.close()
#     pool.join()
#
#
#     tp = sum([x['tp'] for x in results])     # 计算结果列表中所有字典元素中tp（真正例）的总和，并赋值给tp变量
#     fp = sum([x['fp'] for x in results])     # 计算结果列表中所有字典元素中fp（假正例）的总和，并赋值给fp变量
#     fn = sum([x['fn'] for x in results])     # 计算结果列表中所有字典元素中fn（假负例）的总和，并赋值给fn变量
#
#     # 如果tp和fp都等于0，或者tp和fn都等于0，即没有正确或错误的预测
#     if (tp + fp) == 0 or (tp + fn) == 0:
#         return 0
#
#     precision = tp / (tp + fp)     # 计算精确率（precision），即tp除以tp加fp，并赋值给precision变量
#     recall = tp / (tp + fn)     # 计算召回率（recall），即tp除以tp加fn，并赋值给recall变量
#
#     # 如果精确率和召回率都大于0，即有正确的预测
#     if precision > 0 and recall > 0:
#         f1 = (2 * precision * recall) / (precision + recall)   # 计算F1 score，即2乘以精确率乘以召回率除以精确率加召回率，并赋值给f1变量
#     else:
#         f1 = 0
#     return f1
def kuzushiji_f1(sub, solution):
    """
    Calculates the competition metric.
    Args:
        sub: submissions, as a Pandas dataframe
        solution: solution, as a Pandas dataframe
    Returns:
        f1 score
    """
    # 如果提交的预测结果和真实的标注结果的数据框中的image_id列的值不完全相同，即图片编号不匹配，则抛出一个值错误异常，提示图片编号不匹配
    if not all(sub['image_id'].values == solution['image_id'].values):
        raise ValueError("Submission image id codes don't match solution")

    # 使用列表推导式调用score_page函数，传入提交的预测结果和真实的标注结果的数据框中的labels列的值，即预测的标签字符串和真实的标签字符串，返回一个结果列表
    results = [score_page(pred, truth) for pred, truth in zip(sub['labels'].values, solution['labels'].values)]

    tp = sum([x['tp'] for x in results])     # 计算结果列表中所有字典元素中tp（真正例）的总和，并赋值给tp变量
    fp = sum([x['fp'] for x in results])     # 计算结果列表中所有字典元素中fp（假正例）的总和，并赋值给fp变量
    fn = sum([x['fn'] for x in results])     # 计算结果列表中所有字典元素中fn（假负例）的总和，并赋值给fn变量

    # 如果tp和fp都等于0，或者tp和fn都等于0，即没有正确或错误的预测
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0

    precision = tp / (tp + fp)     # 计算精确率（precision），即tp除以tp加fp，并赋值给precision变量
    recall = tp / (tp + fn)     # 计算召回率（recall），即tp除以tp加fn，并赋值给recall变量

    # 如果精确率和召回率都大于0，即有正确的预测
    if precision > 0 and recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)   # 计算F1 score，即2乘以精确率乘以召回率除以精确率加召回率，并赋值给f1变量
    else:
        f1 = 0
    return f1


if __name__ == '__main__':
    # # 如果命令行参数的个数等于1，即只有文件名，没有其他参数, 就会打印使用说明
    if len(sys.argv) == 1: # 打印出使用说明，格式为：python 文件名 [-h] --sub_path SUB_PATH --solution_path SOLUTION_PATH
        print('Usage: python {} [-h] --sub_path SUB_PATH --solution_path SOLUTION_PATH'.format(sys.argv[0]))
        exit()

    # 调用之前定义的函数，获取解析器对象
    parser = define_console_parser()
    # 使用解析器对象解析命令行参数，并返回一个命名空间对象，包含了所有的参数值
    shell_args = parser.parse_args()

    # 使用pandas读取提交的预测结果的文件，返回一个数据框对象
    sub = pd.read_csv(shell_args.sub_path)
    solution = pd.read_csv(shell_args.solution_path)

    """ 排序过程 """
    # 创建一个新的列，把image_id里面的下划线替换成空字符串
    sub['image_id_no_underscore'] = sub['image_id'].str.replace('_', '')

    # 使用sort_values方法，按照image_id_no_underscore的升序排列sub表格，并且不改变原来的索引
    sub_sorted = sub.sort_values(by='image_id_no_underscore', ascending=True, ignore_index=False)

    # 把image_id_no_underscore这个新的列删除
    sub_sorted.drop('image_id_no_underscore', axis=1, inplace=True)
    sub = sub_sorted


    # 将提交的预测结果的数据框中的列名改为image_id和labels，分别表示图片编号和预测的标签字符串
    sub = sub.rename(columns={'rowId': 'image_id', 'PredictionString': 'labels'})
    # 将真实的标注结果的数据框中的列名改为image_id和labels，分别表示图片编号和真实的标签字符串
    solution = solution.rename(columns={'rowId': 'image_id', 'PredictionString': 'labels'})

    # 调用kuzushiji_f1函数，传入提交的预测结果和真实的标注结果的数据框，计算出F1 score，并赋值给score变量
    score = kuzushiji_f1(sub, solution)
    print('F1 score of: {0}'.format(score))
