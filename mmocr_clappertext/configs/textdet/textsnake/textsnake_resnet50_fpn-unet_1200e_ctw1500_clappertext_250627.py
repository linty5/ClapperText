"""ClapperText fine-tuning config for TextSnake (ResNet-50) pre-trained on CTW1500 (paper Table 5)."""

# _base_ = [
#     '_base_textsnake_resnet50_fpn-unet.py',
#     # '../_base_/datasets/ctw1500.py',
#     # '../_base_/default_runtime.py',
#     # '../_base_/schedules/schedule_sgd_1200e.py',
# ]

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

load_from = "https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500/textsnake_resnet50_fpn-unet_1200e_ctw1500_20220825_221459-c0b6adc4.pth"
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
    type="TextSnake",
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        norm_eval=True,
        style="caffe",
    ),
    neck=dict(type="FPN_UNet", in_channels=[256, 512, 1024, 2048], out_channels=32),
    det_head=dict(
        type="TextSnakeHead",
        in_channels=32,
        module_loss=dict(type="TextSnakeModuleLoss"),
        postprocessor=dict(type="TextSnakePostprocessor", text_repr_type="poly"),
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
    dict(type="LoadOCRAnnotations", with_bbox=True, with_polygon=True, with_label=True),
    # dict(
    #     type='TorchVisionWrapper',
    #     op='ColorJitter',
    #     brightness=32.0 / 255,
    #     saturation=0.5),
    dict(
        type="RandomApply",
        transforms=[dict(type="RandomCrop", min_side_ratio=0.3)],
        prob=0.65,
    ),
    # dict(
    #     type='RandomRotate',
    #     max_angle=20,
    #     pad_with_fixed_color=False,
    #     use_canvas=True),
    # dict(
    #     type='BoundedScaleAspectJitter',
    #     long_size_bound=800,
    #     short_size_bound=480,
    #     ratio_range=(0.7, 1.3),
    #     aspect_ratio_range=(0.9, 1.1)),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(type="Resize", scale=800, keep_ratio=True),
                dict(type="SourceImagePad", target_scale=800),
            ],
            dict(type="Resize", scale=800, keep_ratio=False),
        ],
        prob=[0.4, 0.6],
    ),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type="PackTextDetInputs", meta_keys=("img_path", "ori_shape", "img_shape")),
]

test_pipeline = [
    dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
    dict(type="Resize", scale=(1333, 736), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadOCRAnnotations", with_polygon=True, with_bbox=True, with_label=True),
    dict(
        type="PackTextDetInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]


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
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=clappertext_det_train,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=clappertext_det_test,
)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)
