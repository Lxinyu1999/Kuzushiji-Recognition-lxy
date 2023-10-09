default_scope = 'mmdet' # default_scope是指默认的配置域，这里是’mmdet’，表示使用mmdetection的配置。

# 指定了一些默认的训练和测试时的钩子（hook），用来在不同的阶段执行一些操作 。
# hook是一种能够改变程序执行流程的技术，可以在预定义的位置执行预定义的函数。mmdetection和mmcv提供了一些有用的hook，用来在训练和测试时执行一些操作，如打印日志，保存模型，调整学习率等
default_hooks = dict(
    timer=dict(type='IterTimerHook'), # timer是一个IterTimerHook，用来记录每个迭代的时间。
    logger=dict(type='LoggerHook', interval=50), # logger是一个LoggerHook，用来打印日志，这里设置了间隔为50。
    param_scheduler=dict(type='ParamSchedulerHook'), # param_scheduler是一个ParamSchedulerHook，用来调整参数，如学习率等。
    checkpoint=dict(type='CheckpointHook', interval=1), # checkpoint是一个CheckpointHook，用来保存模型，这里设置了间隔为1epoch。
    sampler_seed=dict(type='DistSamplerSeedHook'), # sampler_seed是一个DistSamplerSeedHook，用来设置分布式采样器的随机种子。
    visualization=dict(type='DetVisualizationHook')) # visualization是一个DetVisualizationHook，用来可视化检测结果。

# 指定了一些环境变量，如cudnn_benchmark，mp_cfg，dist_cfg等。
env_cfg = dict(
    cudnn_benchmark=False, # 是否启用cudnn的基准模式。
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # 多进程（multiprocessing）的配置
    dist_cfg=dict(backend='nccl'), # 分布式（distributed）的配置
)

# 可视化的后端。这里只有一个元素，是LocalVisBackend，表示使用本地的可视化后端。
vis_backends = [dict(type='LocalVisBackend')]

# 可视化器的类型和参数。这里使用了DetLocalVisualizer，表示使用本地的检测可视化器。
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# 指定了日志处理器的类型和参数。这里使用了LogProcessor，表示使用默认的日志处理器。window_size和by_epoch是日志处理器的参数。
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO' # 指定了日志的级别。这里设置为’INFO’，表示只打印信息级别以上的日志。
load_from = None # 指定了加载预训练模型的路径。这里设置为None，表示不加载预训练模型。
resume = False # 是否从上次训练中恢复。这里设置为False，表示不从上次训练中恢复。
