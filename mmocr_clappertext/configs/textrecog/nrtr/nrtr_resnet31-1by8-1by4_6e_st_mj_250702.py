"""ClapperText fine-tuning config for NRTR-R31 (1/8) (paper Tables 2 and 3)."""

_base_ = [
    "nrtr_resnet31-1by16-1by8_6e_st_mj_250702.py",
]
randomness = dict(seed=11)

load_from = "https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_resnet31-1by8-1by4_6e_st_mj/nrtr_resnet31-1by8-1by4_6e_st_mj_20220916_103322-a6a2a123.pth"

model = dict(backbone=dict(last_stage_pool=False))
