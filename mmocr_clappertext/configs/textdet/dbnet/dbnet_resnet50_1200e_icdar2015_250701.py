"""ClapperText fine-tuning config for DBNet (ResNet-50) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015_250628.py",
]

load_from = "https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50_1200e_icdar2015/dbnet_resnet50_1200e_icdar2015_20221102_115917-54f50589.pth"

_base_.model.backbone = dict(
    type="mmdet.ResNet",
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type="BN", requires_grad=True),
    norm_eval=True,
    style="pytorch",
    init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
)

_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.002

param_scheduler = [
    dict(type="LinearLR", end=12, start_factor=0.001),
    dict(type="PolyLR", power=0.9, eta_min=1e-7, begin=12, end=36),
]
