optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adadelta', lr=1.0))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0),
]
