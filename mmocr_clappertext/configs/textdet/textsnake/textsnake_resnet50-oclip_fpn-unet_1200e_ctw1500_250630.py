"""ClapperText fine-tuning config for TextSnake (ResNet-50 + OCLIP) pre-trained on CTW1500 (paper Table 5)."""

_base_ = [
    "textsnake_resnet50_fpn-unet_1200e_ctw1500_clappertext_250627.py",
]

load_from = "https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth"

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
