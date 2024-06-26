# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adadelta'))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[3, 4], end=5),
]

