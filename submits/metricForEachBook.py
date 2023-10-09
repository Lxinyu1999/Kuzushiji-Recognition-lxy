import argparse
import multiprocessing
import sys
import csv
import operator
import numpy as np
import pandas as pd


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


    # 读取提交的预测结果和真实的标注结果的CSV文件，分别赋值给sub和solution两个数据框
    sub = pd.read_csv('biaozhu/det_boxes_dino36.csv')
    solution = pd.read_csv('biaozhu/ground_truth_with_id.csv')

    # 定义一个空列表，用于存储每本书的识别指标
    book_scores = []

    # 使用sub数据框的image_id列的apply方法，传入一个lambda函数，将每个图片编号按照第一个下划线分割，取第一个元素作为书名，并赋值给sub数据框的book列
    sub['book'] = sub['image_id'].apply(lambda x: x.split('_')[0])

    # 使用sub数据框的groupby方法，按照book列分组，得到一个分组对象，并赋值给grouped变量
    grouped = sub.groupby('book')

    # 遍历分组对象中的每个元素，将其中的书名和对应的预测结果数据框分别赋值给book和preds变量
    for book, preds in grouped:
        # 使用solution数据框的loc方法，传入一个布尔型数组，筛选出与当前书名相同的图片编号对应的真实标注结果，并赋值给truth变量
        truth = solution.loc[solution['image_id'].str.startswith(book)]

        # 调用kuzushiji_f1函数，传入preds和truth两个数据框作为参数，得到整本书的F1 score，并赋值给f1变量
        f1 = kuzushiji_f1(preds, truth)

        # 使用preds和truth两个数据框的labels列调用score_page函数，得到整本书的tp, fp, fn，并赋值给score变量
        score = score_page(preds['labels'], truth['labels'])

        # 计算整本书的精确率（precision），即tp除以tp加fp，并赋值给precision变量
        precision = score['tp'] / (score['tp'] + score['fp'])

        # 计算整本书的召回率（recall），即tp除以tp加fn，并赋值给recall变量
        recall = score['tp'] / (score['tp'] + score['fn'])

        # 将当前书名和对应的精确率、召回率、F1 score组成一个字典，并添加到book_scores列表中
        book_scores.append({'book': book, 'precision': precision, 'recall': recall, 'f1': f1})

    # 将book_scores列表转换为一个Pandas数据框，并赋值给book_scores_df变量
    book_scores_df = pd.DataFrame(book_scores)

    # 打印book_scores_df数据框，查看每本书的识别指标
    print(book_scores_df)
