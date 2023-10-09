# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/' #这个是默认写法，应该是在linux上的。
data_root = 'D:/mmdetection/mmdetection-main/data/coco/' # 我们按照自己的要求配置了coco数据集和路径

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    # https://mmcv.readthedocs.io/zh_CN/2.x/api/generated/mmcv.transforms.LoadImageFromFile.html
    dict(type='LoadImageFromFile', backend_args=backend_args), # LoadImageFromFile是一个数据预处理的类，从最新版的mmcv中找到. 有一个参数to_float32，如果为True，就会将图片的数值范围从0-255转换为0-1。默认为false

    # https://mmcv.readthedocs.io/zh_CN/2.x/api/generated/mmcv.transforms.LoadAnnotations.html?highlight=loadannotation#mmcv.transforms.LoadAnnotations
    dict(type='LoadAnnotations', with_bbox=True), # 从标注文件中加载目标框的信息，with_bbox是一个布尔值，表示是否加载目标框。


    dict(type='Resize', scale=(1333, 800), keep_ratio=True), # 对图片进行缩放，scale是一个元组，表示缩放后的最大宽度和高度，keep_ratio是一个布尔值，表示是否保持原始的长宽比。
    dict(type='RandomFlip', prob=0.5), # 对图片进行随机翻转，prob是一个浮点数，表示翻转的概率。
    dict(type='PackDetInputs') # 将图片和标注信息打包成一个字典，用于送入模型进行训练。
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline 如果测试集没有ground truth，可删掉该句话
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        # meta_keys是一个元组，表示需要保存的元数据的键名，比如图片的id、路径、原始形状、缩放后的形状和缩放因子等。
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2, # 用于加载数据的子进程数量，这里是2。太大可以调
    persistent_workers=True, # 一个布尔值，表示是否保持子进程的状态，这里是True。
    sampler=dict(type='DefaultSampler', shuffle=True), # 随机打乱是默认为true的
    batch_sampler=dict(type='AspectRatioBatchSampler'), # 这里使用的是AspectRatioBatchSampler类，它会根据图片的长宽比来分组，并尽量保证每个批次的图片具有相似的形状。
    dataset=dict(
        type=dataset_type, # 之前第一行定义的是coco
        data_root=data_root, # 数据集的根目录
        ann_file='annotations/instances_train2017.json', # 遵循coco的写法
        data_prefix=dict(img='train2017/'), # 也是coco的写法
        filter_cfg=dict(filter_empty_gt=True, min_size=32), # 用于过滤图片的配置，这里有两个参数：filter_empty_gt: 一个布尔值，表示是否过滤掉没有标注信息的图片，这里是True。min_size: 一个整数，表示过滤掉小于该尺寸的图片，这里是32。
        pipeline=train_pipeline, # 一个列表，表示数据预处理流程，这里使用的是train_pipeline
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader # 测试集和验证集dataloader的配置是一样的。

val_evaluator = dict(
    type='CocoMetric', # 表示用于评估的类，这里使用的是CocoMetric类，它是基于COCO数据集的评估方法。https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/coco_metric.py
    ann_file=data_root + 'annotations/instances_val2017.json', # 也是coco数据集的写法
    metric='bbox', # 表示评估的指标，这里是’bbox’，表示只评估目标框的性能。
    format_only=False, # 表示是否只输出格式化的结果，而不进行评估，这里是False。
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
