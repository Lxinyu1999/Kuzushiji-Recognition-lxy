_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DeformableDETR',
    num_queries=300, # 表示模型在每张图像上预测的最大目标数是300，即模型输出的边界框和类别的数量。
    num_feature_levels=4, # 表示模型使用的特征层级数是4，即模型从骨干网络和特征提取网络中提取4个不同尺度的特征图，用于编码器和解码器的输入。
    with_box_refine=False, # 表示模型是否使用边界框精调，即在解码器输出边界框后，是否再使用一个额外的网络对边界框进行进一步的优化。
    as_two_stage=False, # 表示模型是否作为两阶段模型，即在解码器输出边界框后，是否再使用一个额外的网络对边界框内的特征进行分类。

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, # 是否将数据从BGR颜色空间转换为RGB颜色空间，因为不同的图像库可能使用不同的颜色空间表示图像。
        pad_size_divisor=0), # 表示是否对数据进行填充，使其能够被指定的数字整除，这样可以方便后续的卷积操作。如果为1，则表示不进行填充。原版为1，试试0

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    # 这段是用来定义特征提取网络的设置的
    neck=dict(
        type='ChannelMapper', # 表示特征提取网络的类型是ChannelMapper，一种用于将不同尺度的特征图映射到相同通道数的网络。
        in_channels=[512, 1024, 2048], # 表示输入的特征图的通道数，分别对应骨干网络的第二、第三、第四层输出的特征图。
        kernel_size=1, # 1×1卷积来减少通道数
        out_channels=256, # 表示输出的特征图的通道数，即将输入的特征图映射到256个通道。
        act_cfg=None, # 表示激活函数的配置，即是否在卷积操作后使用激活函数。如果为None，则表示不使用激活函数。
        # 表示归一化方式的配置，即是否在卷积操作后使用归一化操作。如果为字典，则表示使用指定类型和参数的归一化方式。这里使用了GN（Group Normalization），一种将通道分成若干组进行归一化的方式，其中num_groups表示分组数。
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),

    # 表示编码器的总体设置
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6, # 表示编码器的层数，即编码器由6个相同的层组成。
        # 表示每层的配置
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
    # 表示自注意力机制的配置，即每层中使用的一种用于捕捉特征之间的依赖关系的机制。这里使用了MultiScaleDeformableAttention，
    # 其中embed_dims表示特征向量的维度，batch_first表示是否将批量维度放在第一位。
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            # 表示前馈网络的配置，即每层中使用的一种用于增强特征表达能力的网络。
            # 这里使用了一个两层的全连接网络，其中embed_dims表示输入和输出特征向量的维度，feedforward_channels表示中间层特征向量的维度，ffn_drop表示丢弃率。
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),

    # 表示解码器的总体设置
    decoder=dict(
        num_layers=6, # 表示解码器的层数，即解码器由6个相同的层组成。
        return_intermediate=True, # 表示是否返回中间结果，即是否将每层输出的边界框和类别返回，用于后续的边界框精调或两阶段模型。
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention 表示自注意力机制的配置
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention 表示交叉注意力机制的配置，即每层中使用的一种用于捕捉编码器和解码器之间特征的依赖关系的机制。
                embed_dims=256,
                batch_first=True),
            # ffn_cfg: 表示前馈网络的配置，即每层中使用的一种用于增强特征表达能力的网络。这里使用了一个两层的全连接网络，其中embed_dims表示输入和输出特征向量的维度，
            # feedforward_channels表示中间层特征向量的维度，ffn_drop表示丢弃率。
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=True), # 表示后置归一化方式的配置，即是否在每层输出后使用归一化操作。如果为None，则表示不使用归一化操作。换成True试试


    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            # 这里使用了HungarianAssigner，一种基于匈牙利算法的分配器，用于将预测的边界框和类别与真实的边界框和类别进行匹配，
            # 其中match_costs表示匹配代价函数的类型和权重等。
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# 表示训练时的数据增强流程，包括一系列的数据变换操作，按照顺序依次对每个样本进行处理。
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5), # dict(type=‘RandomFlip’, prob=0.5): 表示随机翻转图像和边界框的操作，其中prob表示翻转的概率。
    dict(
    # dict( type=‘RandomChoice’, transforms=[…]): 表示随机选择一个数据变换列表进行变换的操作，其中transforms表示可选的数据变换列表，每个列表中包含若干个数据变换操作。这里有两个可选的列表：
        type='RandomChoice',
        transforms=[
            [
                dict(
                    # 表示随机选择一个尺寸对图像和边界框进行缩放的操作，其中scales表示可选的尺寸列表，keep_ratio表示是否保持图像的长宽比
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs') # dict(type=‘PackDetInputs’): 表示将图像和标注信息打包成一个字典的操作，方便后续的模型输入。

]
train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# learning policy
max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32) # 表示自动缩放学习率的设置，即一种用于根据批量大小和GPU数目调整学习率的方式。这样可以保持不同训练设置下的相同收敛速度和性能。
