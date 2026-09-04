dictionary = dict(
    type="Dictionary",
    dict_file="{{ fileDirname }}/../../../dicts/english_digits_symbols.txt",
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True,
)

model = dict(
    type="SARNet",
    data_preprocessor=dict(
        type="TextRecogDataPreprocessor", mean=[127, 127, 127], std=[127, 127, 127]
    ),
    backbone=dict(type="ResNet31OCR"),
    encoder=dict(
        type="SAREncoder",
        enc_bi_rnn=False,
        enc_do_rnn=0.1,
        enc_gru=False,
    ),
    decoder=dict(
        type="ParallelSARDecoder",
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True,
        postprocessor=dict(type="AttentionPostprocessor"),
        module_loss=dict(type="CEModuleLoss", ignore_first_char=True, reduction="mean"),
        dictionary=dictionary,
        max_seq_len=30,
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
    dict(
        type="RandomApply",
        prob=0.25,
        transforms=[
            dict(
                type="TorchVisionWrapper",
                op="ColorJitter",
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1,
            ),
        ],
    ),
    dict(
        type="RescaleToHeight", height=48, min_width=48, max_width=160, width_divisor=4
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
        type="RescaleToHeight", height=48, min_width=48, max_width=160, width_divisor=4
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
                    height=48,
                    min_width=48,
                    max_width=160,
                    width_divisor=4,
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
