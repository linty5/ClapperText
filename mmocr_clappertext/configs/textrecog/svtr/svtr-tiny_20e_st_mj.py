_base_ = [
    "_base_svtr-tiny.py",
    # '../_base_/default_runtime.py',
    # '../_base_/datasets/mjsynth.py',
    # '../_base_/datasets/synthtext.py',
    # '../_base_/datasets/cute80.py',
    # '../_base_/datasets/iiit5k.py',
    # '../_base_/datasets/svt.py',
    # '../_base_/datasets/svtp.py',
    # '../_base_/datasets/icdar2013.py',
    # '../_base_/datasets/icdar2015.py',
    # '../_base_/schedules/schedule_adam_base.py',
]

optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=3e-4))
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="LinearLR", end=2, start_factor=0.001, convert_to_iter_based=True),
    dict(type="MultiStepLR", milestones=[18, 27], end=36),
]
default_scope = "mmocr"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=20,
        save_best="ClapperTextCrop/recog/word_acc_ignore_case_symbol",
        rule="greater",
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
        monitor="ClapperTextCrop/recog/word_acc_ignore_case_symbol",
        rule="greater",
        patience=18,
        min_delta=0.0001,
    )
]
# Logging
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=10, by_epoch=True)

# load_from = None
resume = False

# Evaluation
val_evaluator = dict(
    type="MultiDatasetsEvaluator",
    metrics=[
        dict(type="WordMetric", mode=["exact", "ignore_case", "ignore_case_symbol"]),
        dict(type="CharMetric"),
    ],
    dataset_prefixes=["ClapperTextCrop"],
)
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="TextRecogLocalVisualizer", name="visualizer", vis_backends=vis_backends
)

tta_model = dict(type="EncoderDecoderRecognizerTTAModel")


# dataset settings
clappertext_crop_data_root = "data/clappertext"

clappertext_crop_train = dict(
    type="OCRDataset",
    data_root=clappertext_crop_data_root,
    ann_file="textrecog_finetune/train.json",
    data_prefix=dict(img_path="recognition/imgs/train"),
    pipeline=_base_.train_pipeline,
    test_mode=False,
)

clappertext_crop_test = dict(
    type="OCRDataset",
    data_root=clappertext_crop_data_root,
    ann_file="textrecog_finetune/val.json",
    data_prefix=dict(img_path="recognition/imgs/val"),
    pipeline=_base_.test_pipeline,
    test_mode=True,
)

train_dataloader = dict(
    batch_size=128,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=clappertext_crop_train,
)

test_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=clappertext_crop_test,
)

val_dataloader = test_dataloader

# val_evaluator = dict(
#     dataset_prefixes=['ClapperTextCrop'],)
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=128)
