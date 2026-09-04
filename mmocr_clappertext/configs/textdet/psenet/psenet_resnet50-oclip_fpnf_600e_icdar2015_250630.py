"""ClapperText fine-tuning config for PSENet (ResNet-50 + OCLIP) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "psenet_resnet50_fpnf_600e_icdar2015_250628.py",
]

load_from = "https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50-oclip_fpnf_600e_icdar2015/psenet_resnet50-oclip_fpnf_600e_icdar2015_20221101_131357-2bdca389.pth"

_base_.model.backbone = dict(
    type="CLIPResNet",
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/"
        "mmocr/backbone/resnet50-oclip-7ba0c533.pth",
    ),
)

_base_.model.data_preprocessor = dict(
    type="TextDetDataPreprocessor",
    mean=[122.77, 116.23, 104.10],
    std=[68.50, 66.62, 70.32],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)
