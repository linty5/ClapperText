"""ClapperText cropped-word training and validation datasets."""

clappertext_crop_data_root = "data/clappertext"

clappertext_crop_train = dict(
    type="OCRDataset",
    data_root=clappertext_crop_data_root,
    ann_file="textrecog_finetune/train.json",
    data_prefix=dict(img_path="recognition/imgs/train"),
    pipeline=None,
    test_mode=False,
)

clappertext_crop_test = dict(
    type="OCRDataset",
    data_root=clappertext_crop_data_root,
    ann_file="textrecog_finetune/val.json",
    data_prefix=dict(img_path="recognition/imgs/val"),
    pipeline=None,
    test_mode=True,
)
