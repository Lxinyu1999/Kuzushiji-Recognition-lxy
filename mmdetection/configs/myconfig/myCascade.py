# openmmlab有覆盖机制，所以你只需用把修改的属性词典写上就行了。另外，有些变量你要定义，比如backend_args,data_root 等等，这些没法直接从base继承
# 除非你把使用变量的地方比如data_root全部换成路径
_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

#### 1. model settings
model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),


    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32')),

    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256),


    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.2, 0.5, 1.0, 2.0, 5.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),

        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),

    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),


        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4328,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4328,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4328,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),


    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),

        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),


        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler', # 是否修改为OHEMSampler？
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),

            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),

    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=700)))



# dataset settings
# 按照官网要求修改为最新版了 https://mmdetection.readthedocs.io/zh_CN/latest/migration/config_migration.html
# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/' #这个是默认写法，应该是在linux上的。
data_root = 'D:/mmdetection/mmdetection-main/data/coco/' # 我们按照自己的要求配置了coco数据集和路径
backend_args = None

train_pipeline = [
    # https://mmcv.readthedocs.io/zh_CN/2.x/api/generated/mmcv.transforms.LoadImageFromFile.html
    dict(type='LoadImageFromFile', backend_args=backend_args), # LoadImageFromFile是一个数据预处理的类，从最新版的mmcv中找到. 有一个参数to_float32，如果为True，就会将图片的数值范围从0-255转换为0-1。默认为false

    # https://mmcv.readthedocs.io/zh_CN/2.x/api/generated/mmcv.transforms.LoadAnnotations.html?highlight=loadannotation#mmcv.transforms.LoadAnnotations
    dict(type='LoadAnnotations', with_bbox=True), # 从标注文件中加载目标框的信息，with_bbox是一个布尔值，表示是否加载目标框。

dict(
    type='RandomChoiceResize',
    scales= [(512, 512), (544, 544), (576, 576), (608, 608), (640, 640), (672, 672), (704, 704), (736, 736), (768, 768)],
    keep_ratio=True),
    dict(type='RandomFlip', prob=0.5), # 对图片进行随机翻转，prob是一个浮点数，表示翻转的概率。
    dict(type='PackDetInputs') # 将图片和标注信息打包成一个字典，用于送入模型进行训练。
]

"""
最新版的mmdet当中，没有val_pipeline是因为它默认使用test_pipeline来进行验证(后面dataloader能看到)。这样做的原因是为了保持训练和验证的一致性，避免因为不同的数据预处理而导致的性能差异。
如果你想使用不同的val_pipeline，你可以在配置文件中自定义，或者在train.py中修改val_dataset.pipeline的赋值。但是要注意，如果你使用不同的val_pipeline，可能会影响模型的评估结果
"""

test_pipeline = [ # 去掉了resize，按原图输入
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # If you don't have a gt annotation, delete the pipeline 如果测试集没有ground truth，可删掉该句话
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        # meta_keys是一个元组，表示需要保存的元数据的键名，比如图片的id、路径、原始形状、缩放后的形状和缩放因子等。
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 接下来是dataloader，这里就不复制粘贴了，一模一样。注意的是，val_dataloader里面有一句： pipeline=test_pipeline,而 test_dataloader = val_dataloader
# 所以测试集和验证集dataloader的配置是一样的。我们这里重新定义test_dataloader和test_evaluator，但是没有影响val


# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/test.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    # format_only=True，表示只输出格式化的结果，而不进行评估。这是因为测试集通常没有ground truth，所以无法计算评估指标。但是我们有。
    format_only=False,
    ann_file=data_root + 'annotations/test.json',
    outfile_prefix='./work_dirs/test')



# 评测器
"""
在 3.x 版本中，模型精度评测不再与数据集绑定，而是通过评测器（Evaluator）来完成。 
评测器配置分为 val_evaluator 和 test_evaluator 两部分，其中 val_evaluator 用于验证集评测，test_evaluator 用于测试集评测，
对应 2.x 版本中的 evaluation 字段。 下表列出了 2.x 版本与 3.x 版本中的评测器的对应关系
https://mmdetection.readthedocs.io/zh_CN/latest/migration/config_migration.html
"""





#### 3. scheduler

# optimizer

optim_wrapper = dict(
    type='OptimWrapper',
    # 使用了SGD优化器，学习率为0.02，动量为0.9，权重衰减为0.0001
    optimizer=dict(type='SGD', lr=0.02 / 16 * 1, momentum=0.9, weight_decay=0.0001))

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=1.0/3, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=500),  # 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=0,   # 从第一个 epoch 开始
        end=12,  # 到第 12 个 epoch 结束
        milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
]



#### 4. runtime settings
"""
一般来说，mmdetection支持两种预训练的方式：
预训练骨干网络（backbone）：这是指使用在ImageNet等数据集上预训练的骨干网络的权重，作为模型的初始权重。这样可以加速模型的收敛，提高模型的性能。mmdetection提供了多种骨干网络的预训练权重，你可以在配置文件中指定init_cfg参数来使用它们。
预训练整个模型（full model）：这是指使用在COCO等数据集上预训练的整个模型的权重，作为模型的初始权重。这样可以在不同的数据集上进行迁移学习（transfer learning），或者从一个已有的训练结果继续训练。mmdetection提供了多种模型的预训练权重，你可以在模型库（model zoo）中找到它们，并在训练命令中使用--load-from或--resume-from参数来使用它们

如果你只修改了runtime当中的load_from参数，其他的参数都是直接继承base里面的话，
那么你只需要在你自定义的配置文件里面写你修改的参数就可以了，不需要写其他的参数。
"""
# 来源在这儿：https://github.com/HRNet/HRNet-Object-Detection
# 使用Cascade Mask R-CNN的模型预训练权重，不会和Cascade R-CNN冲突，因为它们共享了相同的骨干网络和检测分支。只是在训练时，需要忽略掉分割分支的权重，或者将它们冻结（freeze）。这样可以利用预训练权重提供的特征信息，加速模型的收敛，提高模型的性能。
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_20190810-76f61cd0.pth'
