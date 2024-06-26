_base_ = [
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_sar_resnet31_parallel-decoder.py',
]

# dataset settings
train_list = [_base_.icdar2015_textrecog_train]
test_list = [_base_.icdar2015_textrecog_test]
default_hooks = dict(logger=dict(type='LoggerHook', interval=1))

train_dataloader = dict(
    batch_size=32,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),

    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(dataset_prefixes=['IC15'])
test_evaluator = val_evaluator