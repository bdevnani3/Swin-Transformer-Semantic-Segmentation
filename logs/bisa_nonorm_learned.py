/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:26:58,450 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2: TITAN Xp
GPU 3: TITAN X (Pascal)
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.11) 5.4.0 20160609
PyTorch: 1.11.0
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.5.2 (Git Hash a9302535553c73243c632ad3c4c80beec3d19a1e)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.3
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.2
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 

TorchVision: 0.12.0
OpenCV: 4.5.5
MMCV: 1.5.0
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 11.3
MMSegmentation: 0.11.0+7f98bcf
------------------------------------------------------------

2022-05-22 03:26:58,450 - mmseg - INFO - Distributed training: True
2022-05-22 03:26:58,795 - mmseg - INFO - Config:
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BisaSwinTransformer',
        img_size=512,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        reverse_attention_locations=[1],
        apply_bidirectional_layer_norms=False,
        bidirectional_lambda_value=-100.0,
        embed_dim=96),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=2)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=320000)
checkpoint_config = dict(by_epoch=False, interval=32000)
evaluation = dict(interval=32000, metric='mIoU')
work_dir = './work_dirs/bisa_nonorm_learned'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:27:01,114 - mmseg - INFO - EncoderDecoder(
  (backbone): BisaSwinTransformer(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=96, window_size=(7, 7), num_heads=3
              (activation): GELU()
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=96, window_size=(7, 7), num_heads=3
              (activation): GELU()
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=384, out_features=192, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=192, window_size=(7, 7), num_heads=6
              (activation): GELU()
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (G): Linear(in_features=32, out_features=1024, bias=False)
              (bias_generator): Linear(in_features=32, out_features=32, bias=False)
              (local_proj): Linear(in_features=192, out_features=192, bias=True)
              (global_proj): Linear(in_features=192, out_features=192, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=192, window_size=(7, 7), num_heads=6
              (activation): GELU()
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (G): Linear(in_features=32, out_features=1024, bias=False)
              (bias_generator): Linear(in_features=32, out_features=32, bias=False)
              (local_proj): Linear(in_features=192, out_features=192, bias=True)
              (global_proj): Linear(in_features=192, out_features=192, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=768, out_features=384, bias=False)
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (activation): GELU()
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (activation): GELU()
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (2): SwinTransformerBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (activation): GELU()
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (3): SwinTransformerBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (activation): GELU()
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (4): SwinTransformerBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (activation): GELU()
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (5): SwinTransformerBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (activation): GELU()
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=1536, out_features=768, bias=False)
          (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=768, window_size=(7, 7), num_heads=24
              (activation): GELU()
              (qkv): Linear(in_features=768, out_features=2304, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=768, out_features=768, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): BiDirectionalWindowAttention(
              dim=768, window_size=(7, 7), num_heads=24
              (activation): GELU()
              (qkv): Linear(in_features=768, out_features=2304, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=768, out_features=768, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (orthogonal_loss): L1Loss()
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (norm0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
    (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (decode_head): UPerHead(
    input_transform=multiple_select, ignore_index=255, align_corners=False
    (loss_decode): CrossEntropyLoss()
    (conv_seg): Conv2d(512, 150, kernel_size=(1, 1), stride=(1, 1))
    (dropout): Dropout2d(p=0.1, inplace=False)
    (psp_modules): PPM(
      (0): Sequential(
        (0): AdaptiveAvgPool2d(output_size=1)
        (1): ConvModule(
          (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
      )
      (1): Sequential(
        (0): AdaptiveAvgPool2d(output_size=2)
        (1): ConvModule(
          (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
      )
      (2): Sequential(
        (0): AdaptiveAvgPool2d(output_size=3)
        (1): ConvModule(
          (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
      )
      (3): Sequential(
        (0): AdaptiveAvgPool2d(output_size=6)
        (1): ConvModule(
          (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
      )
    )
    (bottleneck): ConvModule(
      (conv): Conv2d(2816, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activate): ReLU(inplace=True)
    )
    (lateral_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(96, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU()
      )
      (1): ConvModule(
        (conv): Conv2d(192, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU()
      )
      (2): ConvModule(
        (conv): Conv2d(384, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU()
      )
    )
    (fpn_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU()
      )
      (1): ConvModule(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU()
      )
      (2): ConvModule(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU()
      )
    )
    (fpn_bottleneck): ConvModule(
      (conv): Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activate): ReLU(inplace=True)
    )
  )
  (auxiliary_head): FCNHead(
    input_transform=None, ignore_index=255, align_corners=False
    (loss_decode): CrossEntropyLoss()
    (conv_seg): Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
    (dropout): Dropout2d(p=0.1, inplace=False)
    (convs): Sequential(
      (0): ConvModule(
        (conv): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
  )
)
2022-05-22 03:27:01,937 - mmseg - INFO - Loaded 20210 images
2022-05-22 03:27:06,095 - mmseg - INFO - Loaded 2000 images
2022-05-22 03:27:06,095 - mmseg - INFO - Start running, host: bdevnani3@ripl-s1, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_nonorm_learned
2022-05-22 03:27:06,096 - mmseg - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) PolyLrUpdaterHook                  
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) PolyLrUpdaterHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_iter:
(VERY_HIGH   ) PolyLrUpdaterHook                  
(LOW         ) IterTimerHook                      
 -------------------- 
after_train_iter:
(ABOVE_NORMAL) GradientCumulativeOptimizerHook    
(NORMAL      ) CheckpointHook                     
(NORMAL      ) DistEvalHook                       
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) CheckpointHook                     
(NORMAL      ) DistEvalHook                       
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_epoch:
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_epoch:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_run:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
2022-05-22 03:27:06,096 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 03:27:06,096 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_nonorm_learned by HardDiskBackend.
2022-05-22 03:27:20,001 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 03:27:22,294 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
2022-05-22 03:27:54,266 - mmseg - INFO - Iter [50/320000]	lr: 9.799e-07, eta: 2 days, 18:10:31, time: 0.745, data_time: 0.010, memory: 7999, decode.loss_seg: 4.1118, decode.acc_seg: 0.2411, aux.loss_seg: 1.6321, aux.acc_seg: 0.9252, loss: 5.7440
2022-05-22 03:28:27,014 - mmseg - INFO - Iter [100/320000]	lr: 1.979e-06, eta: 2 days, 14:10:58, time: 0.655, data_time: 0.003, memory: 7999, decode.loss_seg: 3.9794, decode.acc_seg: 0.5279, aux.loss_seg: 1.5918, aux.acc_seg: 0.8470, loss: 5.5712
2022-05-22 03:28:59,931 - mmseg - INFO - Iter [150/320000]	lr: 2.979e-06, eta: 2 days, 12:56:42, time: 0.658, data_time: 0.003, memory: 7999, decode.loss_seg: 4.0588, decode.acc_seg: 4.4630, aux.loss_seg: 1.6344, aux.acc_seg: 1.1495, loss: 5.6932
2022-05-22 03:29:32,704 - mmseg - INFO - Iter [200/320000]	lr: 3.978e-06, eta: 2 days, 12:15:30, time: 0.655, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8949, decode.acc_seg: 9.9135, aux.loss_seg: 1.5854, aux.acc_seg: 1.4172, loss: 5.4802
2022-05-22 03:30:05,454 - mmseg - INFO - Iter [250/320000]	lr: 4.976e-06, eta: 2 days, 11:50:04, time: 0.655, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8647, decode.acc_seg: 14.0447, aux.loss_seg: 1.5971, aux.acc_seg: 5.3337, loss: 5.4618
2022-05-22 03:30:38,102 - mmseg - INFO - Iter [300/320000]	lr: 5.974e-06, eta: 2 days, 11:31:06, time: 0.653, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8206, decode.acc_seg: 14.0550, aux.loss_seg: 1.5900, aux.acc_seg: 7.9581, loss: 5.4106
2022-05-22 03:31:10,724 - mmseg - INFO - Iter [350/320000]	lr: 6.972e-06, eta: 2 days, 11:17:00, time: 0.652, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8237, decode.acc_seg: 15.8484, aux.loss_seg: 1.6120, aux.acc_seg: 11.2014, loss: 5.4357
2022-05-22 03:31:43,241 - mmseg - INFO - Iter [400/320000]	lr: 7.970e-06, eta: 2 days, 11:04:55, time: 0.650, data_time: 0.003, memory: 7999, decode.loss_seg: 3.7036, decode.acc_seg: 16.0524, aux.loss_seg: 1.5709, aux.acc_seg: 12.4854, loss: 5.2745
2022-05-22 03:32:15,637 - mmseg - INFO - Iter [450/320000]	lr: 8.967e-06, eta: 2 days, 10:53:57, time: 0.648, data_time: 0.003, memory: 7999, decode.loss_seg: 3.6013, decode.acc_seg: 16.7029, aux.loss_seg: 1.5477, aux.acc_seg: 14.0055, loss: 5.1490
2022-05-22 03:32:48,019 - mmseg - INFO - Iter [500/320000]	lr: 9.964e-06, eta: 2 days, 10:44:55, time: 0.648, data_time: 0.003, memory: 7999, decode.loss_seg: 3.5731, decode.acc_seg: 19.2624, aux.loss_seg: 1.5542, aux.acc_seg: 17.2458, loss: 5.1273
2022-05-22 03:33:20,394 - mmseg - INFO - Iter [550/320000]	lr: 1.096e-05, eta: 2 days, 10:37:22, time: 0.648, data_time: 0.003, memory: 7999, decode.loss_seg: 3.5302, decode.acc_seg: 17.4854, aux.loss_seg: 1.5289, aux.acc_seg: 15.9140, loss: 5.0591
2022-05-22 03:33:52,702 - mmseg - INFO - Iter [600/320000]	lr: 1.196e-05, eta: 2 days, 10:30:23, time: 0.646, data_time: 0.003, memory: 7999, decode.loss_seg: 3.4502, decode.acc_seg: 19.5226, aux.loss_seg: 1.5132, aux.acc_seg: 16.8054, loss: 4.9634
2022-05-22 03:34:24,983 - mmseg - INFO - Iter [650/320000]	lr: 1.295e-05, eta: 2 days, 10:24:11, time: 0.646, data_time: 0.003, memory: 7999, decode.loss_seg: 3.4404, decode.acc_seg: 18.5909, aux.loss_seg: 1.5099, aux.acc_seg: 17.9079, loss: 4.9503
2022-05-22 03:34:57,316 - mmseg - INFO - Iter [700/320000]	lr: 1.395e-05, eta: 2 days, 10:19:11, time: 0.647, data_time: 0.003, memory: 7999, decode.loss_seg: 3.2650, decode.acc_seg: 20.7639, aux.loss_seg: 1.4636, aux.acc_seg: 19.4903, loss: 4.7287
2022-05-22 03:35:29,613 - mmseg - INFO - Iter [750/320000]	lr: 1.494e-05, eta: 2 days, 10:14:31, time: 0.646, data_time: 0.003, memory: 7999, decode.loss_seg: 3.2595, decode.acc_seg: 20.0954, aux.loss_seg: 1.4459, aux.acc_seg: 17.5423, loss: 4.7054
2022-05-22 03:36:01,878 - mmseg - INFO - Iter [800/320000]	lr: 1.594e-05, eta: 2 days, 10:10:09, time: 0.645, data_time: 0.003, memory: 7999, decode.loss_seg: 3.2745, decode.acc_seg: 20.9147, aux.loss_seg: 1.4712, aux.acc_seg: 19.0005, loss: 4.7458
2022-05-22 03:36:34,124 - mmseg - INFO - Iter [850/320000]	lr: 1.693e-05, eta: 2 days, 10:06:07, time: 0.645, data_time: 0.003, memory: 7999, decode.loss_seg: 3.1424, decode.acc_seg: 21.0471, aux.loss_seg: 1.4231, aux.acc_seg: 18.4255, loss: 4.5655
2022-05-22 03:37:06,390 - mmseg - INFO - Iter [900/320000]	lr: 1.793e-05, eta: 2 days, 10:02:36, time: 0.645, data_time: 0.003, memory: 7999, decode.loss_seg: 3.1157, decode.acc_seg: 20.2443, aux.loss_seg: 1.4046, aux.acc_seg: 18.8504, loss: 4.5203
2022-05-22 03:37:38,614 - mmseg - INFO - Iter [950/320000]	lr: 1.892e-05, eta: 2 days, 9:59:09, time: 0.644, data_time: 0.003, memory: 7999, decode.loss_seg: 3.1990, decode.acc_seg: 21.4935, aux.loss_seg: 1.4498, aux.acc_seg: 19.6266, loss: 4.6488
2022-05-22 03:38:10,771 - mmseg - INFO - Exp name: bisa_nonorm_learned.py
2022-05-22 03:38:10,771 - mmseg - INFO - Iter [1000/320000]	lr: 1.992e-05, eta: 2 days, 9:55:39, time: 0.643, data_time: 0.003, memory: 7999, decode.loss_seg: 3.0051, decode.acc_seg: 21.3369, aux.loss_seg: 1.3792, aux.acc_seg: 18.3477, loss: 4.3843
2022-05-22 03:38:42,903 - mmseg - INFO - Iter [1050/320000]	lr: 2.091e-05, eta: 2 days, 9:52:14, time: 0.642, data_time: 0.003, memory: 7999, decode.loss_seg: 2.9945, decode.acc_seg: 22.7095, aux.loss_seg: 1.3727, aux.acc_seg: 19.8937, loss: 4.3672
2022-05-22 03:39:15,025 - mmseg - INFO - Iter [1100/320000]	lr: 2.190e-05, eta: 2 days, 9:49:09, time: 0.643, data_time: 0.003, memory: 7999, decode.loss_seg: 3.0341, decode.acc_seg: 22.6697, aux.loss_seg: 1.3972, aux.acc_seg: 19.2127, loss: 4.4313
2022-05-22 03:39:47,056 - mmseg - INFO - Iter [1150/320000]	lr: 2.290e-05, eta: 2 days, 9:45:49, time: 0.641, data_time: 0.003, memory: 7999, decode.loss_seg: 2.8257, decode.acc_seg: 22.7428, aux.loss_seg: 1.3100, aux.acc_seg: 20.0563, loss: 4.1357
2022-05-22 03:40:19,003 - mmseg - INFO - Iter [1200/320000]	lr: 2.389e-05, eta: 2 days, 9:42:20, time: 0.639, data_time: 0.003, memory: 7999, decode.loss_seg: 2.8229, decode.acc_seg: 25.3002, aux.loss_seg: 1.3298, aux.acc_seg: 20.4463, loss: 4.1526
2022-05-22 03:40:51,008 - mmseg - INFO - Iter [1250/320000]	lr: 2.488e-05, eta: 2 days, 9:39:20, time: 0.640, data_time: 0.003, memory: 7999, decode.loss_seg: 2.9065, decode.acc_seg: 23.1515, aux.loss_seg: 1.3483, aux.acc_seg: 19.1004, loss: 4.2547
2022-05-22 03:41:23,036 - mmseg - INFO - Iter [1300/320000]	lr: 2.587e-05, eta: 2 days, 9:36:38, time: 0.641, data_time: 0.003, memory: 7999, decode.loss_seg: 2.7659, decode.acc_seg: 24.7360, aux.loss_seg: 1.3027, aux.acc_seg: 20.9676, loss: 4.0686
2022-05-22 03:41:55,040 - mmseg - INFO - Iter [1350/320000]	lr: 2.687e-05, eta: 2 days, 9:33:59, time: 0.640, data_time: 0.003, memory: 7999, decode.loss_seg: 2.6854, decode.acc_seg: 23.3541, aux.loss_seg: 1.2614, aux.acc_seg: 19.8802, loss: 3.9468
2022-05-22 03:42:27,023 - mmseg - INFO - Iter [1400/320000]	lr: 2.786e-05, eta: 2 days, 9:31:24, time: 0.640, data_time: 0.004, memory: 7999, decode.loss_seg: 2.6232, decode.acc_seg: 24.2579, aux.loss_seg: 1.2380, aux.acc_seg: 19.8203, loss: 3.8612
2022-05-22 03:42:59,032 - mmseg - INFO - Iter [1450/320000]	lr: 2.885e-05, eta: 2 days, 9:29:04, time: 0.640, data_time: 0.003, memory: 7999, decode.loss_seg: 2.6311, decode.acc_seg: 24.6623, aux.loss_seg: 1.2489, aux.acc_seg: 20.5025, loss: 3.8800
2022-05-22 03:43:31,072 - mmseg - INFO - Iter [1500/320000]	lr: 2.984e-05, eta: 2 days, 9:26:58, time: 0.641, data_time: 0.003, memory: 7999, decode.loss_seg: 2.6001, decode.acc_seg: 25.1422, aux.loss_seg: 1.2257, aux.acc_seg: 21.2087, loss: 3.8259
2022-05-22 03:44:03,080 - mmseg - INFO - Iter [1550/320000]	lr: 3.083e-05, eta: 2 days, 9:24:51, time: 0.640, data_time: 0.003, memory: 7999, decode.loss_seg: 2.6620, decode.acc_seg: 25.8902, aux.loss_seg: 1.2528, aux.acc_seg: 21.1964, loss: 3.9147
2022-05-22 03:44:35,123 - mmseg - INFO - Iter [1600/320000]	lr: 3.182e-05, eta: 2 days, 9:22:57, time: 0.641, data_time: 0.004, memory: 7999, decode.loss_seg: 2.5717, decode.acc_seg: 25.4271, aux.loss_seg: 1.2006, aux.acc_seg: 21.4057, loss: 3.7724
2022-05-22 03:45:07,123 - mmseg - INFO - Iter [1650/320000]	lr: 3.281e-05, eta: 2 days, 9:20:59, time: 0.640, data_time: 0.003, memory: 7999, decode.loss_seg: 2.5699, decode.acc_seg: 24.2543, aux.loss_seg: 1.1973, aux.acc_seg: 21.0374, loss: 3.7672
2022-05-22 03:45:39,111 - mmseg - INFO - Iter [1700/320000]	lr: 3.380e-05, eta: 2 days, 9:19:05, time: 0.640, data_time: 0.003, memory: 7999, decode.loss_seg: 2.5201, decode.acc_seg: 25.7658, aux.loss_seg: 1.1778, aux.acc_seg: 20.7995, loss: 3.6979
2022-05-22 03:46:11,110 - mmseg - INFO - Iter [1750/320000]	lr: 3.479e-05, eta: 2 days, 9:17:17, time: 0.640, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4540, decode.acc_seg: 26.6171, aux.loss_seg: 1.1518, aux.acc_seg: 21.6487, loss: 3.6058
2022-05-22 03:46:43,070 - mmseg - INFO - Iter [1800/320000]	lr: 3.578e-05, eta: 2 days, 9:15:26, time: 0.639, data_time: 0.003, memory: 7999, decode.loss_seg: 2.5226, decode.acc_seg: 25.1975, aux.loss_seg: 1.1765, aux.acc_seg: 19.8303, loss: 3.6991
2022-05-22 03:47:15,147 - mmseg - INFO - Iter [1850/320000]	lr: 3.677e-05, eta: 2 days, 9:14:00, time: 0.642, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4804, decode.acc_seg: 24.7341, aux.loss_seg: 1.1444, aux.acc_seg: 20.2345, loss: 3.6248
2022-05-22 03:47:47,322 - mmseg - INFO - Iter [1900/320000]	lr: 3.775e-05, eta: 2 days, 9:12:53, time: 0.644, data_time: 0.004, memory: 7999, decode.loss_seg: 2.4825, decode.acc_seg: 26.4032, aux.loss_seg: 1.1528, aux.acc_seg: 21.3288, loss: 3.6353
2022-05-22 03:48:19,367 - mmseg - INFO - Iter [1950/320000]	lr: 3.874e-05, eta: 2 days, 9:11:25, time: 0.641, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3092, decode.acc_seg: 27.6049, aux.loss_seg: 1.0805, aux.acc_seg: 22.9886, loss: 3.3897
2022-05-22 03:48:51,447 - mmseg - INFO - Exp name: bisa_nonorm_learned.py
2022-05-22 03:48:51,447 - mmseg - INFO - Iter [2000/320000]	lr: 3.973e-05, eta: 2 days, 9:10:09, time: 0.642, data_time: 0.004, memory: 7999, decode.loss_seg: 2.4200, decode.acc_seg: 26.8565, aux.loss_seg: 1.1138, aux.acc_seg: 22.5746, loss: 3.5339
2022-05-22 03:49:23,514 - mmseg - INFO - Iter [2050/320000]	lr: 4.072e-05, eta: 2 days, 9:08:51, time: 0.641, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4105, decode.acc_seg: 28.0153, aux.loss_seg: 1.1145, aux.acc_seg: 22.8806, loss: 3.5249
2022-05-22 03:49:55,721 - mmseg - INFO - Iter [2100/320000]	lr: 4.170e-05, eta: 2 days, 9:07:57, time: 0.644, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4683, decode.acc_seg: 26.9707, aux.loss_seg: 1.1258, aux.acc_seg: 22.6283, loss: 3.5942
2022-05-22 03:50:27,898 - mmseg - INFO - Iter [2150/320000]	lr: 4.269e-05, eta: 2 days, 9:06:59, time: 0.644, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3531, decode.acc_seg: 26.9017, aux.loss_seg: 1.0779, aux.acc_seg: 23.1873, loss: 3.4310
2022-05-22 03:51:00,226 - mmseg - INFO - Iter [2200/320000]	lr: 4.368e-05, eta: 2 days, 9:06:24, time: 0.647, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3530, decode.acc_seg: 25.6748, aux.loss_seg: 1.0704, aux.acc_seg: 21.0761, loss: 3.4234
