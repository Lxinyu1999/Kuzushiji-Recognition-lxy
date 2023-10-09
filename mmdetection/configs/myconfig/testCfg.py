# base settings
# 不会冲突，如果你在一个自定义的config文件中，首先指定了_base_，然后又写了model字段，那么model字段会覆盖_base_中的model字段。
# 也就是说，你可以在自定义的config文件中修改或者添加一些model的参数，而不影响_base_中的其他参数。这样可以实现灵活的配置修改和复用。
_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings