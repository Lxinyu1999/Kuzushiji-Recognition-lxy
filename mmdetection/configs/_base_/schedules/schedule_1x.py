# training schedule for 1x，其中1x表示12个epoch

# EpochBasedTrainLoop是指基于轮数（epoch）的训练循环1，它是mmdetection中的一个类，用来定义训练的逻辑
# https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1) # 每隔一个轮数（epoch）就进行一次验证（validation）。
val_cfg = dict(type='ValLoop') # ValLoop和TestLoop也是mmdetection中的类，用来定义验证（validation）和测试（test）的循环
test_cfg = dict(type='TestLoop')

# learning rate
# param_scheduler 是一个列表，包含了两个字典，分别定义了在各个iteration和epoch的学习率调度策略。
param_scheduler = [
    dict(
        # 第一个字典是LinearLR，负责iteration上的调度，表示在开始的500个迭代中，学习率从0.001倍线性增加到1倍
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500), # by_epoch=False是指学习率的调整策略是基于迭代次数（iteration）而不是轮数（epoch）。也就是说，学习率会在每个迭代后根据LinearLR的规则进行变化，而不是在每个轮数后。这样可以更细粒度地控制学习率的变化

    dict(
        # 第二个字典是MultiStepLR，负责epoch上的调度，表示在12个轮数中，学习率在第8和第11轮分别乘以0.1
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    # 优化相关的配置现在已全部集成到 optim_wrapper 中，通常包含三个域：optimizer, paramwise_cfg，clip_grad
    # 其他的优化器以及配置详见：https://mmdetection.readthedocs.io/zh_CN/latest/advanced_guides/customize_runtime.html
    # 已经支持了 Pytorch 中实现的所有优化器，要使用这些优化器唯一要做就是修改配置文件中的 optimi_wrapper 中的 optimzer 域
    type='OptimWrapper',
    # 使用了SGD优化器，学习率为0.02，动量为0.9，权重衰减为0.0001
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# 是否根据基础批量大小自动缩放学习率2。这里设置为False，表示不启用自动缩放学习率。
auto_scale_lr = dict(enable=False, base_batch_size=16)
