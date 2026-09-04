_base_ = [
    "svtr-tiny_20e_st_mj.py",
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-small_20e_st_mj/svtr-small_20e_st_mj-35d800d6.pth"

model = dict(
    encoder=dict(
        embed_dims=[96, 192, 256],
        depth=[3, 6, 6],
        num_heads=[3, 6, 8],
        mixer_types=["Local"] * 8 + ["Global"] * 7,
    )
)
