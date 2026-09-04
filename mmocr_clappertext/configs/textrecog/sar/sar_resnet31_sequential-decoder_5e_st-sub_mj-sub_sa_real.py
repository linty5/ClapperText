_base_ = [
    "sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real.py",
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/sar/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real_20220915_185451-1fd6b1fc.pth"

model = dict(decoder=dict(type="SequentialSARDecoder"))
