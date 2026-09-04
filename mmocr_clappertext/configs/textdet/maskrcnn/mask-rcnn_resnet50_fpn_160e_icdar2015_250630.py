"""ClapperText fine-tuning config for Mask R-CNN (ResNet-50) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "_base_mask-rcnn_resnet50_fpn_250630.py",
    # '../_base_/datasets/icdar2015.py',
    "../_base_/default_runtime.py",
    # '../_base_/schedules/schedule_sgd_base.py',
]
load_from = "https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_icdar2015/mask-rcnn_resnet50_fpn_160e_icdar2015_20220826_154808-ff5c30bf.pth"
# optimizer
# optim_wrapper = dict(optimizer=dict(lr=0.08))
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.08, momentum=0.9, weight_decay=0.0001),
)

train_cfg = dict(max_epochs=36)
# learning policy
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", end=18, start_factor=0.001, by_epoch=False),
    dict(type="MultiStepLR", milestones=[18, 27], end=36),
]

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", interval=10, save_best="icdar/hmean", rule="greater"
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffer=dict(type="SyncBuffersHook"),
    visualization=dict(
        type="VisualizationHook",
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False,
    ),
)

custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="icdar/hmean",
        rule="greater",
        patience=12,
        min_delta=0.0001,
    )
]
# # dataset settings
# icdar2015_textdet_train = _base_.icdar2015_textdet_train
# icdar2015_textdet_test = _base_.icdar2015_textdet_test
# icdar2015_textdet_train.pipeline = _base_.train_pipeline
# icdar2015_textdet_test.pipeline = _base_.test_pipeline

# train_dataloader = dict(
#     batch_size=8,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=icdar2015_textdet_train)

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=icdar2015_textdet_test)

# test_dataloader = val_dataloader

clappertext_det_data_root = "data/clappertext"

clappertext_det_train = dict(
    type="OCRDataset",
    data_root=clappertext_det_data_root,
    ann_file="textdet_finetune/train.json",
    data_prefix=dict(img_path="detection/imgs/train"),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=_base_.train_pipeline,
)

clappertext_det_test = dict(
    type="OCRDataset",
    data_root=clappertext_det_data_root,
    ann_file="textdet_finetune/val.json",
    data_prefix=dict(img_path="detection/imgs/val"),
    test_mode=True,
    pipeline=_base_.test_pipeline,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=clappertext_det_train,
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=clappertext_det_test,
)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)
