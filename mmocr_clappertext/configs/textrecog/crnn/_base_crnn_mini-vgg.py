dictionary = dict(
    type="Dictionary",
    dict_file="{{ fileDirname }}/../../../dicts/lower_english_digits.txt",
    with_padding=True,
    # Ignore symbols outside the pretrained alphanumeric output vocabulary.
    with_unknown=True,
    unknown_token=None,
)

model = dict(
    type="CRNN",
    preprocessor=None,
    backbone=dict(type="MiniVGG", leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(
        type="CRNNDecoder",
        in_channels=512,
        rnn_flag=True,
        module_loss=dict(type="CTCModuleLoss", letter_case="lower"),
        postprocessor=dict(type="CTCPostProcessor"),
        dictionary=dictionary,
    ),
    data_preprocessor=dict(type="TextRecogDataPreprocessor", mean=[127], std=[127]),
)

train_pipeline = [
    dict(
        type="LoadImageFromFile", color_type="grayscale", ignore_empty=True, min_size=2
    ),
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(type="Resize", scale=(100, 32), keep_ratio=False),
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
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(
        type="RescaleToHeight",
        height=32,
        min_width=32,
        max_width=None,
        width_divisor=16,
    ),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

tta_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
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
                    max_width=None,
                    width_divisor=16,
                )
            ],
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
