_base_ = [
    # '../_base_/datasets/mjsynth.py',
    # '../_base_/datasets/synthtext.py',
    # '../_base_/datasets/synthtext_add.py',
    # '../_base_/datasets/coco_text_v1.py',
    # '../_base_/datasets/cute80.py',
    # '../_base_/datasets/iiit5k.py',
    # '../_base_/datasets/svt.py',
    # '../_base_/datasets/svtp.py',
    # '../_base_/datasets/icdar2011.py',
    # '../_base_/datasets/icdar2013.py',
    # '../_base_/datasets/icdar2015.py',
    "../_base_/default_runtime.py",
    # '../_base_/schedules/schedule_adam_step_5e.py',
    "_base_robustscanner_resnet31.py",
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/robust_scanner/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real_20220915_152447-7fc35929.pth"

optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=3e-4))
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="LinearLR", end=2, start_factor=0.001, convert_to_iter_based=True),
    dict(type="MultiStepLR", milestones=[18, 27], end=36),
]

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
test_evaluator = _base_.val_evaluator

auto_scale_lr = dict(base_batch_size=128 * 4)
