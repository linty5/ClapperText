"""Augmentation-ablation config for NRTR-R31 (1/16) on ClapperText: color jitter disabled, 36-epoch schedule (paper setting) (paper Table 4)."""

load_from = "https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_resnet31-1by16-1by8_6e_st_mj/nrtr_resnet31-1by16-1by8_6e_st_mj_20220920_143358-43767036.pth"

default_scope = "mmocr"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=11)

# ClapperTextCrop/recog/word_acc_ignore_case_symbol

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
        patience=30,
        min_delta=0.0001,
    )
]

# Logging
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=10, by_epoch=True)

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
dictionary = dict(
    type="Dictionary",
    dict_file="{{ fileDirname }}/../../../dicts/english_digits_symbols.txt",
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True,
)

model = dict(
    type="NRTR",
    backbone=dict(
        type="ResNet31OCR",
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
        last_stage_pool=True,
    ),
    encoder=dict(type="NRTREncoder"),
    decoder=dict(
        type="NRTRDecoder",
        module_loss=dict(type="CEModuleLoss", ignore_first_char=True, flatten=True),
        postprocessor=dict(type="AttentionPostprocessor"),
        dictionary=dictionary,
        max_seq_len=30,
    ),
    data_preprocessor=dict(
        type="TextRecogDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", ignore_empty=True, min_size=2),
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(
        type="RandomApply",
        prob=0.5,
        transforms=[
            dict(
                type="RandomChoice",
                transforms=[
                    dict(
                        type="RandomRotate",
                        max_angle=15,
                    ),
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomAffine",
                        degrees=15,
                        translate=(0.3, 0.3),
                        scale=(0.5, 2.0),
                        shear=(-45, 45),
                    ),
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomPerspective",
                        distortion_scale=0.5,
                        p=1,
                    ),
                ],
            )
        ],
    ),
    dict(
        type="RandomApply",
        prob=0.25,
        transforms=[
            dict(type="PyramidRescale"),
            # dict(
            #     type='mmdet.Albu',
            #     transforms=[
            #         dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
            #         dict(type='MotionBlur', blur_limit=7, p=0.5),
            #     ]),
        ],
    ),
    # dict(
    #     type='RandomApply',
    #     prob=0.25,
    #     transforms=[
    #         dict(
    #             type='TorchVisionWrapper',
    #             op='ColorJitter',
    #             brightness=0.5,
    #             saturation=0.5,
    #             contrast=0.5,
    #             hue=0.1),
    #     ]),
    dict(
        type="RescaleToHeight", height=32, min_width=32, max_width=160, width_divisor=4
    ),
    dict(type="PadToWidth", width=160),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RescaleToHeight", height=32, min_width=32, max_width=160, width_divisor=16
    ),
    dict(type="PadToWidth", width=160),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

tta_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(
                    type="ConditionApply",
                    true_transforms=[
                        dict(
                            type="ImgAugWrapper",
                            args=[dict(cls="Rot90", k=0, keep_size=False)],
                        )
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                ),
                dict(
                    type="ConditionApply",
                    true_transforms=[
                        dict(
                            type="ImgAugWrapper",
                            args=[dict(cls="Rot90", k=1, keep_size=False)],
                        )
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                ),
                dict(
                    type="ConditionApply",
                    true_transforms=[
                        dict(
                            type="ImgAugWrapper",
                            args=[dict(cls="Rot90", k=3, keep_size=False)],
                        )
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                ),
            ],
            [
                dict(
                    type="RescaleToHeight",
                    height=32,
                    min_width=32,
                    max_width=160,
                    width_divisor=16,
                )
            ],
            [dict(type="PadToWidth", width=160)],
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            [dict(type="LoadOCRAnnotations", with_text=True)],
            [
                dict(
                    type="PackTextRecogInputs",
                    meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
                )
            ],
        ],
    ),
]


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
    pipeline=train_pipeline,
    test_mode=False,
)

clappertext_crop_test = dict(
    type="OCRDataset",
    data_root=clappertext_crop_data_root,
    ann_file="textrecog_finetune/val.json",
    data_prefix=dict(img_path="recognition/imgs/val"),
    pipeline=test_pipeline,
    test_mode=True,
)

train_dataloader = dict(
    batch_size=256,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=clappertext_crop_train,
)

test_dataloader = dict(
    batch_size=256,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=clappertext_crop_test,
)

val_dataloader = test_dataloader

test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=256)
