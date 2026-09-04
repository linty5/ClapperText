_base_ = [
    "svtr-tiny_20e_st_mj.py",
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-base_20e_st_mj/svtr-base_20e_st_mj-ea500101.pth"

model = dict(
    preprocessor=dict(
        output_image_size=(48, 160),
    ),
    encoder=dict(
        img_size=[48, 160],
        max_seq_len=40,
        out_channels=256,
        embed_dims=[128, 256, 384],
        depth=[3, 6, 9],
        num_heads=[4, 8, 12],
        mixer_types=["Local"] * 8 + ["Global"] * 10,
    ),
    decoder=dict(in_channels=256),
)

# train_dataloader = dict(batch_size=256, )
