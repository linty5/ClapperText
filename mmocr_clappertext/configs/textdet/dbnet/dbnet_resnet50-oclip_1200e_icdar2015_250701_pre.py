"""ClapperText fine-tuning config for DBNet (ResNet-50 + OCLIP) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015_250628.py",
]

load_from = "https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth"

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


_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.002

param_scheduler = [
    dict(type="LinearLR", end=12, start_factor=0.001),
    dict(type="PolyLR", power=0.9, eta_min=1e-7, begin=12, end=36),
]
