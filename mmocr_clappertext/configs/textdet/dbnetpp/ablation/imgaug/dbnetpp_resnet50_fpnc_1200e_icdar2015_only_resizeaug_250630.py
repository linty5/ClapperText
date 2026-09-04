"""Augmentation-ablation config for DBNet++ (ResNet-50) on ClapperText: multi-scale resize augmentation only (paper Sec. 4.3)."""

default_scope = "mmocr"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=11)

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

# Logging
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=10, by_epoch=True)

load_from = "https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth"
resume = False

# Evaluation
val_evaluator = dict(type="HmeanIOUMetric")
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="TextDetLocalVisualizer", name="visualizer", vis_backends=vis_backends
)

model = dict(
    type="DBNet",
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPNC",
        in_channels=[256, 512, 1024, 2048],
        lateral_channels=256,
        asf_cfg=dict(attention_type="ScaleChannelSpatial"),
    ),
    det_head=dict(
        type="DBHead",
        in_channels=256,
        module_loss=dict(type="DBModuleLoss"),
        postprocessor=dict(
            type="DBPostprocessor", text_repr_type="quad", epsilon_ratio=0.002
        ),
    ),
    data_preprocessor=dict(
        type="TextDetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
    dict(
        type="LoadOCRAnnotations",
        with_bbox=True,
        with_polygon=True,
        with_label=True,
    ),
    dict(
        type="TorchVisionWrapper",
        op="ColorJitter",
        brightness=32.0 / 255,
        saturation=0.5,
    ),
    dict(
        type="ImgAugWrapper",
        args=[
            # ['Fliplr', 0.5]
            # dict(cls='Affine', rotate=[-10, 10])
            ["Resize", [0.5, 3.0]]
        ],
    ),
    dict(type="RandomCrop", min_side_ratio=0.1),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="Pad", size=(640, 640)),
    dict(type="PackTextDetInputs", meta_keys=("img_path", "ori_shape", "img_shape")),
]

test_pipeline = [
    dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
    dict(type="Resize", scale=(4068, 1024), keep_ratio=True),
    dict(
        type="LoadOCRAnnotations",
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type="PackTextDetInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor", "instances"),
    ),
]

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.003, momentum=0.9, weight_decay=0.0001),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="LinearLR", end=12, start_factor=0.001),
    dict(type="PolyLR", power=0.9, eta_min=1e-7, begin=20, end=36),
]


# dataset settings

clappertext_det_data_root = "data/clappertext"


clappertext_det_train = dict(
    type="OCRDataset",
    data_root=clappertext_det_data_root,
    ann_file="textdet_finetune/train.json",
    data_prefix=dict(img_path="detection/imgs/train"),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
)

clappertext_det_test = dict(
    type="OCRDataset",
    data_root=clappertext_det_data_root,
    ann_file="textdet_finetune/val.json",
    data_prefix=dict(img_path="detection/imgs/val"),
    test_mode=True,
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=16,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=clappertext_det_train,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=clappertext_det_test,
)


test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)
