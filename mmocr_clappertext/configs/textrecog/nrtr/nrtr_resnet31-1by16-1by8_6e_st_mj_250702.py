"""ClapperText fine-tuning config for NRTR-R31 (1/16) (paper Tables 2 and 3)."""

_base_ = [
    # '../_base_/datasets/mjsynth.py',
    # '../_base_/datasets/synthtext.py',
    # '../_base_/datasets/cute80.py',
    # '../_base_/datasets/iiit5k.py',
    # '../_base_/datasets/svt.py',
    # '../_base_/datasets/svtp.py',
    # '../_base_/datasets/icdar2013.py',
    # '../_base_/datasets/icdar2015.py',
    "../_base_/default_runtime.py",
    # '../_base_/schedules/schedule_adam_base.py',
    "_base_nrtr_resnet31_250702.py",
]

randomness = dict(seed=11)

load_from = "https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_resnet31-1by16-1by8_6e_st_mj/nrtr_resnet31-1by16-1by8_6e_st_mj_20220920_143358-43767036.pth"

# optimizer
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=3e-4))
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="MultiStepLR", milestones=[18, 27], end=36),
]

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

test_evaluator = _base_.val_evaluator

auto_scale_lr = dict(base_batch_size=128)
