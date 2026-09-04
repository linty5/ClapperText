"""ClapperText fine-tuning config for PSENet (ResNet-50) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "_base_psenet_resnet50_fpnf.py",
    # '../_base_/datasets/icdar2015.py',
    "../_base_/default_runtime.py",
    # '../_base_/schedules/schedule_adam_600e.py',
]

load_from = "https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50_fpnf_600e_icdar2015/psenet_resnet50_fpnf_600e_icdar2015_20220825_222709-b6741ec3.pth"

# optimizer
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
# optimizer
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=1e-3))
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning rate
param_scheduler = [
    dict(type="PolyLR", power=0.9, end=36),
]

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

# use quadrilaterals for icdar2015
model = dict(
    backbone=dict(style="pytorch"),
    det_head=dict(postprocessor=dict(text_repr_type="quad")),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=clappertext_det_train,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=clappertext_det_test,
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="HmeanIOUMetric", pred_score_thrs=dict(start=0.3, stop=1, step=0.05)
)
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=8)
