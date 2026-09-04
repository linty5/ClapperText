default_scope = "mmocr"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=20,
        save_best="ClapperTextCrop/recog/word_acc_ignore_case_symbol",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffer=dict(type="SyncBuffersHook"),
    visualization=dict(
        type="VisualizationHook",
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False,
    ),
)
custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="ClapperTextCrop/recog/word_acc_ignore_case_symbol",
        rule="greater",
        patience=18,
        min_delta=0.0001,
    )
]
# Logging
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=10, by_epoch=True)

load_from = None
resume = False

# Evaluation
val_evaluator = dict(
    type="MultiDatasetsEvaluator",
    metrics=[
        dict(type="WordMetric", mode=["exact", "ignore_case", "ignore_case_symbol"]),
        dict(type="CharMetric"),
    ],
    dataset_prefixes=["ClapperTextCrop"],
)
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="TextRecogLocalVisualizer", name="visualizer", vis_backends=vis_backends
)

tta_model = dict(type="EncoderDecoderRecognizerTTAModel")
