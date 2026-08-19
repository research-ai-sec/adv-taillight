

optimizer = dict(type='SGD', lr=0.2, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=0.002)
momentum_config = None


runner = dict(type='EpochBasedRunner', max_epochs=150)
