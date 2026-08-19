

optimizer = dict(type='Adam', lr=0.001, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)
momentum_config = None


runner = dict(type='EpochBasedRunner', max_epochs=50)
