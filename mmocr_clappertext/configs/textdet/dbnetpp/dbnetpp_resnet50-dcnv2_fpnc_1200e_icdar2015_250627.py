"""ClapperText fine-tuning config for DBNet++ (ResNet-50 + DCN) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "_base_dbnetpp_resnet50-dcnv2_fpnc.py",
    "../_base_/default_runtime.py",
    # '../_base_/datasets/icdar2015.py',
    # '../_base_/schedules/schedule_sgd_1200e.py',
]

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.007, momentum=0.9, weight_decay=0.0001),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="PolyLR", power=0.9, eta_min=1e-7, end=36),
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

load_from = "https://download.openmmlab.com/mmocr/textdet/dbnetpp/tmp_1.0_pretrain/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-352fec8a.pth"  # noqa

# dataset settings

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
