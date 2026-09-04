"""ClapperText fine-tuning config for FCENet (ResNet-50 + OCLIP) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "fcenet_resnet50_fpn_1500e_icdar2015_250630.py",
]
load_from = "https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015/fcenet_resnet50-oclip_fpn_1500e_icdar2015_20221101_150145-5a6fc412.pth"

_base_.model.backbone = dict(
    type="CLIPResNet",
    out_indices=(1, 2, 3),
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

_base_.train_dataloader.batch_size = 8
_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.0005
