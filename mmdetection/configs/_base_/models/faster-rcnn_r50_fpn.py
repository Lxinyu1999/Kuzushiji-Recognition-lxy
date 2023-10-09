# model settings
# 详见：https://mmdetection.readthedocs.io/zh_CN/v2.21.0/tutorials/config.html
model = dict(
    type='FasterRCNN', # 检测模型的名称

    # 预处理现已放到模型里面，不在coco_detection里了。
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),

    # backbone网络，使用的是ResNet50
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4, # resnet的stage数量
        out_indices=(0, 1, 2, 3), # 输出的stage的序号
        frozen_stages=1, # 冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        norm_cfg=dict(type='BN', requires_grad=True), #  归一化层(norm layer)的配置项。requires_grad表示是否训练归一化里的 gamma 和 beta。
        norm_eval=True, # 是否冻结 BN 里的统计项。
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), # 加载通过 ImageNet 预训练的模型

    # 检测器的 neck 是 FPN
    neck=dict(
        type='FPN', # 更多细节可以参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py#L10。
        in_channels=[256, 512, 1024, 2048], # 输入通道数，这与主干网络的输出通道一致
        out_channels=256, # 金字塔特征图每一层的输出通道,与原论文相同
        num_outs=5), # 输出的特征层的数量，也就是原论文中的”P2-P6“

    # 检测头，用的是RPNHead，我们也支持 'GARPNHead' 等。
    rpn_head=dict( # 更多细节可以参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/rpn_head.py#L12。
        type='RPNHead',
        in_channels=256, # 每个输入特征图的输入通道，这与上面 neck 的输出通道一致。
        feat_channels=256, # head 卷积层的特征通道数。
        # RPN当中锚框(Anchor)生成器的配置。
        anchor_generator=dict(
            type='AnchorGenerator', # 大多是方法使用 AnchorGenerator 作为锚点生成器, SSD 检测器使用 `SSDAnchorGenerator`。更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10。
            scales=[8], #  锚点的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            ratios=[0.5, 1.0, 2.0], # 高度和宽度之间的比率。
            strides=[4, 8, 16, 32, 64]), # 锚生成器的步幅。这与 FPN 特征步幅一致。 如果未设置 base_sizes，则当前步幅值将被视为 base_sizes。
        # 在训练和测试期间对框进行编码和解码。
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder', # 框编码器的类别，'DeltaXYWHBBoxCoder' 是最常用的，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9。
            target_means=[.0, .0, .0, .0], # 用于编码和解码框的目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]), # 用于编码和解码框的标准方差
        # 分类分支的损失函数配置，也支持 FocalLoss 等。
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), # RPN通常进行二分类，所以通常使用sigmoid函数。loss_weight表示分类分支的损失权重。
        # 回归分支的损失函数配置。
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)), # 损失类型，我们还支持许多 IoU Losses 和 Smooth L1-loss 等，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56。

    # RoIHead 封装了两步(two-stage)/级联(cascade)检测器当中的第二步，也就是RoI pooling/Align操作。
    roi_head=dict(
        type='StandardRoIHead',  # RoI head 的类型，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/standard_roi_head.py#L10。
        bbox_roi_extractor=dict( # 用于 bbox 回归的 RoI 特征提取器。
            type='SingleRoIExtractor', # RoI 特征提取器的类型，大多数方法使用  SingleRoIExtractor，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level.py#L10。
            # RoI 层的配置
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0), # RoI 层的类别, 也支持 DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/roi_align/roi_align.py#L79
            out_channels=256, # 提取特征的输出通道。
            featmap_strides=[4, 8, 16, 32]), # 多尺度特征图的步幅，应该与主干的架构保持一致。
        # RoIHead 中 box head 的配置.
        bbox_head=dict(
            type='Shared2FCBBoxHead', # bbox head 的类别，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L177。
            in_channels=256, # bbox head 的输入通道。 这与 roi_extractor 中的 out_channels 一致。
            fc_out_channels=1024, # FC 层的输出特征通道
            roi_feat_size=7, # 候选区域(Region of Interest)特征的大小，经典7×7
            num_classes=80, # 分类的类别数量。和coco相对应，如果是自己的数据集需要你自行修改！
            # 第二阶段使用的框编码器。
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder', # 框编码器的类别，大多数情况使用 'DeltaXYWHBBoxCoder'。
                target_means=[0., 0., 0., 0.], # 用于编码和解码框的均值
                target_stds=[0.1, 0.1, 0.2, 0.2]), # 编码和解码的标准方差。因为框更准确，所以值更小，常规设置是 [0.1, 0.1, 0.2, 0.2]。
            reg_class_agnostic=False, # 回归是否与类别无关。
            # 分类分支的损失函数配置，同上
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # 回归分支的损失函数配置，同上
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),


    # model training and testing settings  rpn 和 rcnn 训练超参数的配置
    train_cfg=dict(
        # rpn 的训练配置
        rpn=dict(
            # 分配器(assigner)的配置
            assigner=dict(
                type='MaxIoUAssigner', # 分配器的类型，MaxIoUAssigner 用于许多常见的检测器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10。
                pos_iou_thr=0.7, # IoU >= 0.7(阈值) 被视为正样本。
                neg_iou_thr=0.3, # IoU < 0.3(阈值) 被视为负样本。
                min_pos_iou=0.3, # 将框作为正样本的最小 IoU 阈值。
                match_low_quality=True, # 是否匹配低质量的框(更多细节见 API 文档).
                ignore_iof_thr=-1), # 忽略 bbox 的 IoF 阈值。
            # 正/负采样器(sampler)的配置
            sampler=dict(
                type='RandomSampler', # 采样器类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8。
                num=256, # 样本数量。
                pos_fraction=0.5, # 正样本占总样本的比例。
                neg_pos_ub=-1, # 基于正样本数量的负样本上限。
                add_gt_as_proposals=False),  # 采样后是否添加 GT 作为 proposal。
            allowed_border=-1, # 填充有效锚点后允许的边框。
            pos_weight=-1, # 训练期间正样本的权重。
            debug=False), # 是否设置调试(debug)模式

        # 在训练期间生成 proposals 的配置
        rpn_proposal=dict(
            nms_pre=2000, # NMS 前的 box 数
            max_per_img=1000, # NMS 后要保留的 box 数量。
            # NMS 的配置
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0), # 允许的最小 box 尺寸

        # roi head 的配置。
        rcnn=dict(
            # 第二阶段分配器的配置，这与 rpn 中的不同
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,  # IoU >= 0.5(阈值)被认为是正样本。
                neg_iou_thr=0.5,  # IoU < 0.5(阈值)被认为是负样本。
                min_pos_iou=0.5,  # 将 box 作为正样本的最小 IoU 阈值
                match_low_quality=False, # 是否匹配低质量下的 box(有关更多详细信息，请参阅 API 文档)。
                ignore_iof_thr=-1), # 忽略 bbox 的 IoF 阈值
            sampler=dict(
                type='RandomSampler', #采样器的类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8。
                num=512,  # 样本数量
                pos_fraction=0.25, # 正样本占总样本的比例。
                neg_pos_ub=-1, # 基于正样本数量的负样本上限。
                add_gt_as_proposals=True), # 采样后是否添加 GT 作为 proposal。
            pos_weight=-1, # 训练期间正样本的权重。
            debug=False)), # 是否设置调试模式。

    # 用于测试 rnn 和 rnn 超参数的配置
    test_cfg=dict(
        # 测试阶段生成 proposals 的配置
        rpn=dict(
            nms_pre=1000, # NMS 前的 box 数
            max_per_img=1000,   # NMS 后要保留的 box 数量
            # NMS 的配置
            nms=dict(type='nms', iou_threshold=0.7), # NMS 阈值设置为0.7
            min_bbox_size=0), # box 允许的最小尺寸
        # roi heads 的配置
        rcnn=dict(
            score_thr=0.05, # bbox 的分数阈值
            nms=dict(type='nms', iou_threshold=0.5), # 第二步的 NMS 配置
            max_per_img=100) # 每张图像的最大检测次数
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
