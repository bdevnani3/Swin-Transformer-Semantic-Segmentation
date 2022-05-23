_base_ = [
    "../_base_/models/upernet_bisa_swin_transformer.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_320k.py",
]


model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        reverse_attention_locations=[1],
        drop_path_rate=0.3,
        patch_norm=True,
        apply_bidirectional_layer_norms= False, # no norm
        bidirectional_lambda_value=-100.0, # lambda = 0
        lambda_learned=False,

    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150),
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=2)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=3000,
    # TODO(): figure out what this is doing
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
