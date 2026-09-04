"""ClapperText fine-tuning config for DBNet++ (ResNet-50) pre-trained on ICDAR 2015 (paper Table 5)."""

_base_ = [
    "dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015_250627.py",
]

randomness = dict(seed=11)

load_from = "https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth"
resume = False

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
_base_.optim_wrapper.optimizer.lr = 0.003

param_scheduler = [
    dict(type="LinearLR", end=12, start_factor=0.001),
    dict(type="PolyLR", power=0.9, eta_min=1e-7, begin=12, end=36),
]
