/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/torch/package/_directory_reader.py:17: UserWarning: Failed to initialize NumPy: numpy.core.multiarray failed to import (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:68.)
  _dtype_to_storage = {data_type(0).dtype: data_type for data_type in _storages}
Traceback (most recent call last):
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/runpy.py", line 184, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/runpy.py", line 110, in _get_module_details
    __import__(pkg_name)
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/torch/__init__.py", line 721, in <module>
    import torch.utils.data
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/torch/utils/data/__init__.py", line 38, in <module>
    from torch.utils.data.dataloader_experimental import DataLoader2
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/torch/utils/data/dataloader_experimental.py", line 11, in <module>
    from torch.utils.data.datapipes.iter import IterableWrapper
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/torch/utils/data/datapipes/__init__.py", line 1, in <module>
    from . import iter
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/torch/utils/data/datapipes/iter/__init__.py", line 37, in <module>
    from torch.utils.data.datapipes.iter.selecting import (
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/torch/utils/data/datapipes/iter/selecting.py", line 7, in <module>
    import pandas  # type: ignore[import]
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/pandas/__init__.py", line 22, in <module>
    from pandas.compat.numpy import (
  File "/nethome/bdevnani3/anaconda3/lib/python3.8/site-packages/pandas/compat/numpy/__init__.py", line 9, in <module>
    _np_version = np.__version__
AttributeError: module 'numpy' has no attribute '__version__'
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 02:53:37,001 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609
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
MMSegmentation: 0.11.0+250e912
------------------------------------------------------------

2022-05-22 02:53:37,002 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 02:53:37,312 - mmseg - INFO - Config:
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
        bidirectional_lambda_value=0.0,
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
work_dir = './work_dirs/bisa_nonorm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 02:53:39,416 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 02:53:41,796 - mmseg - INFO - Loaded 20210 images
2022-05-22 02:53:45,216 - mmseg - INFO - Loaded 2000 images
2022-05-22 02:53:45,216 - mmseg - INFO - Start running, host: bdevnani3@jill, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_nonorm_0-5
2022-05-22 02:53:45,216 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 02:53:45,217 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 02:53:45,217 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_nonorm_0-5 by HardDiskBackend.
2022-05-22 02:53:58,789 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 02:54:00,785 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
2022-05-22 02:54:29,655 - mmseg - INFO - Iter [50/320000]	lr: 9.799e-07, eta: 2 days, 11:45:59, time: 0.672, data_time: 0.012, memory: 7999, decode.loss_seg: 4.2275, decode.acc_seg: 0.2135, aux.loss_seg: 1.6832, aux.acc_seg: 0.3526, loss: 5.9106
2022-05-22 02:54:59,084 - mmseg - INFO - Iter [100/320000]	lr: 1.979e-06, eta: 2 days, 8:01:50, time: 0.589, data_time: 0.003, memory: 7999, decode.loss_seg: 3.9722, decode.acc_seg: 0.5082, aux.loss_seg: 1.5890, aux.acc_seg: 0.3151, loss: 5.5612
2022-05-22 02:55:28,624 - mmseg - INFO - Iter [150/320000]	lr: 2.979e-06, eta: 2 days, 6:50:39, time: 0.591, data_time: 0.003, memory: 7999, decode.loss_seg: 4.0870, decode.acc_seg: 3.3837, aux.loss_seg: 1.6469, aux.acc_seg: 0.7425, loss: 5.7339
2022-05-22 02:55:58,296 - mmseg - INFO - Iter [200/320000]	lr: 3.978e-06, eta: 2 days, 6:18:18, time: 0.593, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8684, decode.acc_seg: 9.0290, aux.loss_seg: 1.5741, aux.acc_seg: 1.8549, loss: 5.4425
2022-05-22 02:56:27,813 - mmseg - INFO - Iter [250/320000]	lr: 4.976e-06, eta: 2 days, 5:55:26, time: 0.590, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8463, decode.acc_seg: 13.3740, aux.loss_seg: 1.5871, aux.acc_seg: 5.3363, loss: 5.4334
2022-05-22 02:56:57,280 - mmseg - INFO - Iter [300/320000]	lr: 5.974e-06, eta: 2 days, 5:39:08, time: 0.589, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8023, decode.acc_seg: 13.5154, aux.loss_seg: 1.5789, aux.acc_seg: 7.4607, loss: 5.3812
2022-05-22 02:57:26,674 - mmseg - INFO - Iter [350/320000]	lr: 6.972e-06, eta: 2 days, 5:26:16, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 3.8016, decode.acc_seg: 16.5394, aux.loss_seg: 1.6041, aux.acc_seg: 9.7433, loss: 5.4058
2022-05-22 02:57:55,983 - mmseg - INFO - Iter [400/320000]	lr: 7.970e-06, eta: 2 days, 5:15:19, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 3.6323, decode.acc_seg: 17.7082, aux.loss_seg: 1.5466, aux.acc_seg: 10.4542, loss: 5.1790
2022-05-22 02:58:25,408 - mmseg - INFO - Iter [450/320000]	lr: 8.967e-06, eta: 2 days, 5:08:05, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 3.4991, decode.acc_seg: 18.0309, aux.loss_seg: 1.5078, aux.acc_seg: 12.0108, loss: 5.0069
2022-05-22 02:58:54,722 - mmseg - INFO - Iter [500/320000]	lr: 9.964e-06, eta: 2 days, 5:01:00, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 3.5840, decode.acc_seg: 18.9152, aux.loss_seg: 1.5552, aux.acc_seg: 14.0499, loss: 5.1392
2022-05-22 02:59:23,827 - mmseg - INFO - Iter [550/320000]	lr: 1.096e-05, eta: 2 days, 4:53:08, time: 0.582, data_time: 0.004, memory: 7999, decode.loss_seg: 3.5064, decode.acc_seg: 18.8437, aux.loss_seg: 1.5256, aux.acc_seg: 12.9664, loss: 5.0320
2022-05-22 02:59:53,028 - mmseg - INFO - Iter [600/320000]	lr: 1.196e-05, eta: 2 days, 4:47:18, time: 0.584, data_time: 0.003, memory: 7999, decode.loss_seg: 3.4387, decode.acc_seg: 19.1192, aux.loss_seg: 1.5003, aux.acc_seg: 13.6610, loss: 4.9390
2022-05-22 03:00:22,310 - mmseg - INFO - Iter [650/320000]	lr: 1.295e-05, eta: 2 days, 4:42:59, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 3.4649, decode.acc_seg: 19.7811, aux.loss_seg: 1.5209, aux.acc_seg: 15.6372, loss: 4.9858
2022-05-22 03:00:51,394 - mmseg - INFO - Iter [700/320000]	lr: 1.395e-05, eta: 2 days, 4:37:42, time: 0.582, data_time: 0.003, memory: 7999, decode.loss_seg: 3.3865, decode.acc_seg: 20.1298, aux.loss_seg: 1.4998, aux.acc_seg: 16.3876, loss: 4.8864
2022-05-22 03:01:20,406 - mmseg - INFO - Iter [750/320000]	lr: 1.494e-05, eta: 2 days, 4:32:32, time: 0.580, data_time: 0.003, memory: 7999, decode.loss_seg: 3.3184, decode.acc_seg: 20.3329, aux.loss_seg: 1.4740, aux.acc_seg: 16.1425, loss: 4.7924
2022-05-22 03:01:49,517 - mmseg - INFO - Iter [800/320000]	lr: 1.594e-05, eta: 2 days, 4:28:38, time: 0.582, data_time: 0.003, memory: 7999, decode.loss_seg: 3.3018, decode.acc_seg: 22.4563, aux.loss_seg: 1.4836, aux.acc_seg: 17.3774, loss: 4.7854
2022-05-22 03:02:18,531 - mmseg - INFO - Iter [850/320000]	lr: 1.693e-05, eta: 2 days, 4:24:32, time: 0.580, data_time: 0.003, memory: 7999, decode.loss_seg: 3.1112, decode.acc_seg: 22.1426, aux.loss_seg: 1.4167, aux.acc_seg: 17.8388, loss: 4.5279
2022-05-22 03:02:47,618 - mmseg - INFO - Iter [900/320000]	lr: 1.793e-05, eta: 2 days, 4:21:15, time: 0.582, data_time: 0.003, memory: 7999, decode.loss_seg: 3.1061, decode.acc_seg: 21.0719, aux.loss_seg: 1.4059, aux.acc_seg: 16.6760, loss: 4.5120
2022-05-22 03:03:16,554 - mmseg - INFO - Iter [950/320000]	lr: 1.892e-05, eta: 2 days, 4:17:24, time: 0.579, data_time: 0.003, memory: 7999, decode.loss_seg: 3.0724, decode.acc_seg: 22.2480, aux.loss_seg: 1.4042, aux.acc_seg: 17.7784, loss: 4.4766
2022-05-22 03:03:45,618 - mmseg - INFO - Exp name: bisa_nonorm_0-5.py
2022-05-22 03:03:45,618 - mmseg - INFO - Iter [1000/320000]	lr: 1.992e-05, eta: 2 days, 4:14:35, time: 0.581, data_time: 0.003, memory: 7999, decode.loss_seg: 3.0059, decode.acc_seg: 21.6024, aux.loss_seg: 1.3825, aux.acc_seg: 16.7224, loss: 4.3884
2022-05-22 03:04:14,510 - mmseg - INFO - Iter [1050/320000]	lr: 2.091e-05, eta: 2 days, 4:11:08, time: 0.578, data_time: 0.003, memory: 7999, decode.loss_seg: 3.0055, decode.acc_seg: 23.2003, aux.loss_seg: 1.3901, aux.acc_seg: 18.3129, loss: 4.3956
2022-05-22 03:04:43,470 - mmseg - INFO - Iter [1100/320000]	lr: 2.190e-05, eta: 2 days, 4:08:12, time: 0.579, data_time: 0.003, memory: 7999, decode.loss_seg: 2.9578, decode.acc_seg: 23.7218, aux.loss_seg: 1.3771, aux.acc_seg: 18.4655, loss: 4.3349
2022-05-22 03:05:12,330 - mmseg - INFO - Iter [1150/320000]	lr: 2.290e-05, eta: 2 days, 4:05:09, time: 0.577, data_time: 0.004, memory: 7999, decode.loss_seg: 2.8191, decode.acc_seg: 23.1689, aux.loss_seg: 1.3209, aux.acc_seg: 18.7796, loss: 4.1400
2022-05-22 03:05:41,186 - mmseg - INFO - Iter [1200/320000]	lr: 2.389e-05, eta: 2 days, 4:02:13, time: 0.577, data_time: 0.003, memory: 7999, decode.loss_seg: 2.8781, decode.acc_seg: 24.0643, aux.loss_seg: 1.3596, aux.acc_seg: 18.8575, loss: 4.2377
2022-05-22 03:06:10,044 - mmseg - INFO - Iter [1250/320000]	lr: 2.488e-05, eta: 2 days, 3:59:30, time: 0.577, data_time: 0.004, memory: 7999, decode.loss_seg: 2.8660, decode.acc_seg: 23.7229, aux.loss_seg: 1.3511, aux.acc_seg: 18.6784, loss: 4.2171
2022-05-22 03:06:39,113 - mmseg - INFO - Iter [1300/320000]	lr: 2.587e-05, eta: 2 days, 3:57:49, time: 0.581, data_time: 0.003, memory: 7999, decode.loss_seg: 2.7293, decode.acc_seg: 23.9108, aux.loss_seg: 1.2869, aux.acc_seg: 19.3352, loss: 4.0162
2022-05-22 03:07:08,031 - mmseg - INFO - Iter [1350/320000]	lr: 2.687e-05, eta: 2 days, 3:55:39, time: 0.578, data_time: 0.004, memory: 7999, decode.loss_seg: 2.6104, decode.acc_seg: 22.4525, aux.loss_seg: 1.2360, aux.acc_seg: 18.2381, loss: 3.8464
2022-05-22 03:07:36,947 - mmseg - INFO - Iter [1400/320000]	lr: 2.786e-05, eta: 2 days, 3:53:34, time: 0.578, data_time: 0.004, memory: 7999, decode.loss_seg: 2.6384, decode.acc_seg: 24.9650, aux.loss_seg: 1.2596, aux.acc_seg: 19.0691, loss: 3.8980
2022-05-22 03:08:05,794 - mmseg - INFO - Iter [1450/320000]	lr: 2.885e-05, eta: 2 days, 3:51:21, time: 0.577, data_time: 0.004, memory: 7999, decode.loss_seg: 2.7034, decode.acc_seg: 26.1011, aux.loss_seg: 1.2916, aux.acc_seg: 18.8825, loss: 3.9950
2022-05-22 03:08:34,836 - mmseg - INFO - Iter [1500/320000]	lr: 2.984e-05, eta: 2 days, 3:49:57, time: 0.581, data_time: 0.003, memory: 7999, decode.loss_seg: 2.6081, decode.acc_seg: 24.7506, aux.loss_seg: 1.2366, aux.acc_seg: 20.5787, loss: 3.8447
2022-05-22 03:09:03,641 - mmseg - INFO - Iter [1550/320000]	lr: 3.083e-05, eta: 2 days, 3:47:47, time: 0.576, data_time: 0.003, memory: 7999, decode.loss_seg: 2.6007, decode.acc_seg: 24.9041, aux.loss_seg: 1.2335, aux.acc_seg: 19.8910, loss: 3.8343
2022-05-22 03:09:32,459 - mmseg - INFO - Iter [1600/320000]	lr: 3.182e-05, eta: 2 days, 3:45:46, time: 0.576, data_time: 0.003, memory: 7999, decode.loss_seg: 2.5461, decode.acc_seg: 26.3726, aux.loss_seg: 1.2057, aux.acc_seg: 21.5685, loss: 3.7518
2022-05-22 03:10:01,297 - mmseg - INFO - Iter [1650/320000]	lr: 3.281e-05, eta: 2 days, 3:43:55, time: 0.577, data_time: 0.003, memory: 7999, decode.loss_seg: 2.5939, decode.acc_seg: 25.0958, aux.loss_seg: 1.2223, aux.acc_seg: 19.7296, loss: 3.8162
2022-05-22 03:10:30,164 - mmseg - INFO - Iter [1700/320000]	lr: 3.380e-05, eta: 2 days, 3:42:14, time: 0.577, data_time: 0.004, memory: 7999, decode.loss_seg: 2.5419, decode.acc_seg: 27.1411, aux.loss_seg: 1.2013, aux.acc_seg: 21.8346, loss: 3.7432
2022-05-22 03:10:59,111 - mmseg - INFO - Iter [1750/320000]	lr: 3.479e-05, eta: 2 days, 3:40:52, time: 0.579, data_time: 0.004, memory: 7999, decode.loss_seg: 2.4864, decode.acc_seg: 28.0167, aux.loss_seg: 1.1875, aux.acc_seg: 22.7110, loss: 3.6739
2022-05-22 03:11:27,930 - mmseg - INFO - Iter [1800/320000]	lr: 3.578e-05, eta: 2 days, 3:39:10, time: 0.576, data_time: 0.003, memory: 7999, decode.loss_seg: 2.5144, decode.acc_seg: 27.0567, aux.loss_seg: 1.1944, aux.acc_seg: 21.8557, loss: 3.7088
2022-05-22 03:11:56,842 - mmseg - INFO - Iter [1850/320000]	lr: 3.677e-05, eta: 2 days, 3:37:47, time: 0.578, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4468, decode.acc_seg: 25.8407, aux.loss_seg: 1.1439, aux.acc_seg: 21.1403, loss: 3.5908
2022-05-22 03:12:25,838 - mmseg - INFO - Iter [1900/320000]	lr: 3.775e-05, eta: 2 days, 3:36:42, time: 0.580, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4387, decode.acc_seg: 26.2847, aux.loss_seg: 1.1415, aux.acc_seg: 21.7614, loss: 3.5802
2022-05-22 03:12:54,800 - mmseg - INFO - Iter [1950/320000]	lr: 3.874e-05, eta: 2 days, 3:35:33, time: 0.579, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3817, decode.acc_seg: 27.5271, aux.loss_seg: 1.1150, aux.acc_seg: 23.2062, loss: 3.4967
2022-05-22 03:13:23,658 - mmseg - INFO - Exp name: bisa_nonorm_0-5.py
2022-05-22 03:13:23,658 - mmseg - INFO - Iter [2000/320000]	lr: 3.973e-05, eta: 2 days, 3:34:10, time: 0.577, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4370, decode.acc_seg: 27.1313, aux.loss_seg: 1.1289, aux.acc_seg: 21.8174, loss: 3.5658
2022-05-22 03:13:52,662 - mmseg - INFO - Iter [2050/320000]	lr: 4.072e-05, eta: 2 days, 3:33:12, time: 0.580, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3547, decode.acc_seg: 28.2829, aux.loss_seg: 1.0974, aux.acc_seg: 22.8308, loss: 3.4521
2022-05-22 03:14:21,752 - mmseg - INFO - Iter [2100/320000]	lr: 4.170e-05, eta: 2 days, 3:32:28, time: 0.582, data_time: 0.003, memory: 7999, decode.loss_seg: 2.4900, decode.acc_seg: 26.9348, aux.loss_seg: 1.1365, aux.acc_seg: 22.3109, loss: 3.6265
2022-05-22 03:14:50,759 - mmseg - INFO - Iter [2150/320000]	lr: 4.269e-05, eta: 2 days, 3:31:33, time: 0.580, data_time: 0.004, memory: 7999, decode.loss_seg: 2.3216, decode.acc_seg: 27.0446, aux.loss_seg: 1.0639, aux.acc_seg: 23.4589, loss: 3.3855
2022-05-22 03:15:19,730 - mmseg - INFO - Iter [2200/320000]	lr: 4.368e-05, eta: 2 days, 3:30:33, time: 0.579, data_time: 0.004, memory: 7999, decode.loss_seg: 2.3447, decode.acc_seg: 26.6434, aux.loss_seg: 1.0731, aux.acc_seg: 22.2141, loss: 3.4178
2022-05-22 03:15:48,714 - mmseg - INFO - Iter [2250/320000]	lr: 4.466e-05, eta: 2 days, 3:29:37, time: 0.580, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3745, decode.acc_seg: 28.3444, aux.loss_seg: 1.0908, aux.acc_seg: 23.7446, loss: 3.4653
2022-05-22 03:16:17,840 - mmseg - INFO - Iter [2300/320000]	lr: 4.565e-05, eta: 2 days, 3:29:02, time: 0.583, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3049, decode.acc_seg: 27.9749, aux.loss_seg: 1.0608, aux.acc_seg: 23.6722, loss: 3.3657
2022-05-22 03:16:46,939 - mmseg - INFO - Iter [2350/320000]	lr: 4.664e-05, eta: 2 days, 3:28:23, time: 0.582, data_time: 0.003, memory: 7999, decode.loss_seg: 2.2666, decode.acc_seg: 29.8405, aux.loss_seg: 1.0352, aux.acc_seg: 25.3613, loss: 3.3018
2022-05-22 03:17:16,127 - mmseg - INFO - Iter [2400/320000]	lr: 4.762e-05, eta: 2 days, 3:27:57, time: 0.584, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3348, decode.acc_seg: 28.0697, aux.loss_seg: 1.0617, aux.acc_seg: 22.9466, loss: 3.3965
2022-05-22 03:17:45,234 - mmseg - INFO - Iter [2450/320000]	lr: 4.861e-05, eta: 2 days, 3:27:19, time: 0.582, data_time: 0.004, memory: 7999, decode.loss_seg: 2.2493, decode.acc_seg: 28.9098, aux.loss_seg: 1.0257, aux.acc_seg: 24.0874, loss: 3.2750
2022-05-22 03:18:14,319 - mmseg - INFO - Iter [2500/320000]	lr: 4.959e-05, eta: 2 days, 3:26:40, time: 0.582, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3293, decode.acc_seg: 27.8390, aux.loss_seg: 1.0479, aux.acc_seg: 24.0961, loss: 3.3772
2022-05-22 03:18:57,972 - mmseg - INFO - Iter [2550/320000]	lr: 5.057e-05, eta: 2 days, 3:56:14, time: 0.873, data_time: 0.274, memory: 7999, decode.loss_seg: 2.2309, decode.acc_seg: 28.7302, aux.loss_seg: 1.0101, aux.acc_seg: 24.6637, loss: 3.2409
2022-05-22 03:19:27,194 - mmseg - INFO - Iter [2600/320000]	lr: 5.156e-05, eta: 2 days, 3:55:17, time: 0.584, data_time: 0.004, memory: 7999, decode.loss_seg: 2.3350, decode.acc_seg: 30.1557, aux.loss_seg: 1.0589, aux.acc_seg: 25.8375, loss: 3.3939
2022-05-22 03:19:56,591 - mmseg - INFO - Iter [2650/320000]	lr: 5.254e-05, eta: 2 days, 3:54:42, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 2.2835, decode.acc_seg: 27.6575, aux.loss_seg: 1.0284, aux.acc_seg: 23.7410, loss: 3.3118
2022-05-22 03:20:25,906 - mmseg - INFO - Iter [2700/320000]	lr: 5.352e-05, eta: 2 days, 3:53:57, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 2.2952, decode.acc_seg: 29.7310, aux.loss_seg: 1.0376, aux.acc_seg: 24.4976, loss: 3.3328
2022-05-22 03:20:55,176 - mmseg - INFO - Iter [2750/320000]	lr: 5.451e-05, eta: 2 days, 3:53:06, time: 0.585, data_time: 0.004, memory: 7999, decode.loss_seg: 2.3436, decode.acc_seg: 27.5853, aux.loss_seg: 1.0411, aux.acc_seg: 24.7149, loss: 3.3848
2022-05-22 03:21:24,417 - mmseg - INFO - Iter [2800/320000]	lr: 5.549e-05, eta: 2 days, 3:52:16, time: 0.585, data_time: 0.004, memory: 7999, decode.loss_seg: 2.2697, decode.acc_seg: 28.9921, aux.loss_seg: 1.0147, aux.acc_seg: 25.0960, loss: 3.2844
2022-05-22 03:21:53,748 - mmseg - INFO - Iter [2850/320000]	lr: 5.647e-05, eta: 2 days, 3:51:35, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.3214, decode.acc_seg: 28.6963, aux.loss_seg: 1.0378, aux.acc_seg: 25.1312, loss: 3.3592
2022-05-22 03:22:23,107 - mmseg - INFO - Iter [2900/320000]	lr: 5.745e-05, eta: 2 days, 3:50:57, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.2649, decode.acc_seg: 28.1988, aux.loss_seg: 1.0120, aux.acc_seg: 24.1433, loss: 3.2770
2022-05-22 03:22:52,418 - mmseg - INFO - Iter [2950/320000]	lr: 5.844e-05, eta: 2 days, 3:50:15, time: 0.586, data_time: 0.004, memory: 7999, decode.loss_seg: 2.2686, decode.acc_seg: 30.3370, aux.loss_seg: 1.0183, aux.acc_seg: 27.3147, loss: 3.2869
2022-05-22 03:23:21,582 - mmseg - INFO - Exp name: bisa_nonorm_0-5.py
2022-05-22 03:23:21,583 - mmseg - INFO - Iter [3000/320000]	lr: 5.942e-05, eta: 2 days, 3:49:17, time: 0.583, data_time: 0.004, memory: 7999, decode.loss_seg: 2.1896, decode.acc_seg: 30.3199, aux.loss_seg: 0.9887, aux.acc_seg: 27.0654, loss: 3.1783
2022-05-22 03:23:50,789 - mmseg - INFO - Iter [3050/320000]	lr: 5.943e-05, eta: 2 days, 3:48:25, time: 0.584, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1887, decode.acc_seg: 30.3216, aux.loss_seg: 0.9841, aux.acc_seg: 26.6260, loss: 3.1728
2022-05-22 03:24:20,113 - mmseg - INFO - Iter [3100/320000]	lr: 5.942e-05, eta: 2 days, 3:47:45, time: 0.586, data_time: 0.004, memory: 7999, decode.loss_seg: 2.1706, decode.acc_seg: 29.5985, aux.loss_seg: 0.9853, aux.acc_seg: 24.9511, loss: 3.1559
2022-05-22 03:24:49,465 - mmseg - INFO - Iter [3150/320000]	lr: 5.941e-05, eta: 2 days, 3:47:09, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.1865, decode.acc_seg: 28.7668, aux.loss_seg: 0.9875, aux.acc_seg: 25.2314, loss: 3.1740
2022-05-22 03:25:18,811 - mmseg - INFO - Iter [3200/320000]	lr: 5.940e-05, eta: 2 days, 3:46:33, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1365, decode.acc_seg: 30.8846, aux.loss_seg: 0.9662, aux.acc_seg: 26.9244, loss: 3.1026
2022-05-22 03:25:48,201 - mmseg - INFO - Iter [3250/320000]	lr: 5.939e-05, eta: 2 days, 3:46:00, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 2.2533, decode.acc_seg: 29.5554, aux.loss_seg: 1.0083, aux.acc_seg: 25.7584, loss: 3.2617
2022-05-22 03:26:17,601 - mmseg - INFO - Iter [3300/320000]	lr: 5.938e-05, eta: 2 days, 3:45:29, time: 0.588, data_time: 0.004, memory: 7999, decode.loss_seg: 2.1934, decode.acc_seg: 29.3662, aux.loss_seg: 0.9815, aux.acc_seg: 25.1354, loss: 3.1749
2022-05-22 03:26:46,994 - mmseg - INFO - Iter [3350/320000]	lr: 5.937e-05, eta: 2 days, 3:44:57, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1386, decode.acc_seg: 30.6138, aux.loss_seg: 0.9679, aux.acc_seg: 26.5443, loss: 3.1065
2022-05-22 03:27:16,315 - mmseg - INFO - Iter [3400/320000]	lr: 5.936e-05, eta: 2 days, 3:44:19, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 2.2345, decode.acc_seg: 29.2896, aux.loss_seg: 0.9998, aux.acc_seg: 25.9263, loss: 3.2343
2022-05-22 03:27:45,664 - mmseg - INFO - Iter [3450/320000]	lr: 5.935e-05, eta: 2 days, 3:43:43, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.2151, decode.acc_seg: 31.3701, aux.loss_seg: 1.0022, aux.acc_seg: 27.0881, loss: 3.2173
2022-05-22 03:28:14,966 - mmseg - INFO - Iter [3500/320000]	lr: 5.934e-05, eta: 2 days, 3:43:04, time: 0.586, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0531, decode.acc_seg: 32.6727, aux.loss_seg: 0.9322, aux.acc_seg: 27.6464, loss: 2.9853
2022-05-22 03:28:44,391 - mmseg - INFO - Iter [3550/320000]	lr: 5.933e-05, eta: 2 days, 3:42:35, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1221, decode.acc_seg: 33.2701, aux.loss_seg: 0.9723, aux.acc_seg: 28.7727, loss: 3.0944
2022-05-22 03:29:13,724 - mmseg - INFO - Iter [3600/320000]	lr: 5.933e-05, eta: 2 days, 3:41:59, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.2016, decode.acc_seg: 30.4380, aux.loss_seg: 0.9882, aux.acc_seg: 25.9887, loss: 3.1899
2022-05-22 03:29:43,107 - mmseg - INFO - Iter [3650/320000]	lr: 5.932e-05, eta: 2 days, 3:41:27, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1045, decode.acc_seg: 33.4661, aux.loss_seg: 0.9567, aux.acc_seg: 29.0807, loss: 3.0613
2022-05-22 03:30:12,435 - mmseg - INFO - Iter [3700/320000]	lr: 5.931e-05, eta: 2 days, 3:40:50, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1749, decode.acc_seg: 30.6838, aux.loss_seg: 0.9786, aux.acc_seg: 26.4891, loss: 3.1536
2022-05-22 03:30:41,800 - mmseg - INFO - Iter [3750/320000]	lr: 5.930e-05, eta: 2 days, 3:40:17, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0765, decode.acc_seg: 32.1912, aux.loss_seg: 0.9404, aux.acc_seg: 27.4927, loss: 3.0168
2022-05-22 03:31:11,125 - mmseg - INFO - Iter [3800/320000]	lr: 5.929e-05, eta: 2 days, 3:39:41, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0701, decode.acc_seg: 29.8656, aux.loss_seg: 0.9276, aux.acc_seg: 26.4476, loss: 2.9978
2022-05-22 03:31:40,441 - mmseg - INFO - Iter [3850/320000]	lr: 5.928e-05, eta: 2 days, 3:39:04, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1235, decode.acc_seg: 32.0696, aux.loss_seg: 0.9686, aux.acc_seg: 26.9134, loss: 3.0921
2022-05-22 03:32:09,747 - mmseg - INFO - Iter [3900/320000]	lr: 5.927e-05, eta: 2 days, 3:38:26, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 2.1058, decode.acc_seg: 32.3540, aux.loss_seg: 0.9569, aux.acc_seg: 27.7794, loss: 3.0627
2022-05-22 03:32:39,157 - mmseg - INFO - Iter [3950/320000]	lr: 5.926e-05, eta: 2 days, 3:37:57, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9720, decode.acc_seg: 32.5436, aux.loss_seg: 0.9016, aux.acc_seg: 28.5067, loss: 2.8736
2022-05-22 03:33:08,497 - mmseg - INFO - Exp name: bisa_nonorm_0-5.py
2022-05-22 03:33:08,498 - mmseg - INFO - Iter [4000/320000]	lr: 5.925e-05, eta: 2 days, 3:37:22, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0267, decode.acc_seg: 33.9525, aux.loss_seg: 0.9323, aux.acc_seg: 28.4433, loss: 2.9590
2022-05-22 03:33:37,840 - mmseg - INFO - Iter [4050/320000]	lr: 5.924e-05, eta: 2 days, 3:36:48, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9904, decode.acc_seg: 33.6536, aux.loss_seg: 0.9029, aux.acc_seg: 28.8275, loss: 2.8934
2022-05-22 03:34:07,220 - mmseg - INFO - Iter [4100/320000]	lr: 5.923e-05, eta: 2 days, 3:36:15, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.0751, decode.acc_seg: 32.5437, aux.loss_seg: 0.9389, aux.acc_seg: 27.3888, loss: 3.0140
2022-05-22 03:34:36,579 - mmseg - INFO - Iter [4150/320000]	lr: 5.922e-05, eta: 2 days, 3:35:44, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0651, decode.acc_seg: 34.5811, aux.loss_seg: 0.9459, aux.acc_seg: 29.0591, loss: 3.0110
2022-05-22 03:35:05,776 - mmseg - INFO - Iter [4200/320000]	lr: 5.921e-05, eta: 2 days, 3:34:59, time: 0.584, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9784, decode.acc_seg: 35.0951, aux.loss_seg: 0.9121, aux.acc_seg: 29.8582, loss: 2.8906
2022-05-22 03:35:35,117 - mmseg - INFO - Iter [4250/320000]	lr: 5.920e-05, eta: 2 days, 3:34:25, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.0079, decode.acc_seg: 34.9346, aux.loss_seg: 0.9274, aux.acc_seg: 29.7165, loss: 2.9352
2022-05-22 03:36:04,524 - mmseg - INFO - Iter [4300/320000]	lr: 5.919e-05, eta: 2 days, 3:33:56, time: 0.588, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9252, decode.acc_seg: 34.0969, aux.loss_seg: 0.8921, aux.acc_seg: 29.6071, loss: 2.8173
2022-05-22 03:36:33,816 - mmseg - INFO - Iter [4350/320000]	lr: 5.918e-05, eta: 2 days, 3:33:18, time: 0.586, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0874, decode.acc_seg: 34.4524, aux.loss_seg: 0.9584, aux.acc_seg: 29.0819, loss: 3.0458
2022-05-22 03:37:03,065 - mmseg - INFO - Iter [4400/320000]	lr: 5.918e-05, eta: 2 days, 3:32:38, time: 0.585, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9212, decode.acc_seg: 34.6381, aux.loss_seg: 0.8862, aux.acc_seg: 30.1845, loss: 2.8074
2022-05-22 03:37:32,439 - mmseg - INFO - Iter [4450/320000]	lr: 5.917e-05, eta: 2 days, 3:32:07, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.0254, decode.acc_seg: 34.8138, aux.loss_seg: 0.9325, aux.acc_seg: 29.8423, loss: 2.9580
2022-05-22 03:38:01,787 - mmseg - INFO - Iter [4500/320000]	lr: 5.916e-05, eta: 2 days, 3:31:34, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9396, decode.acc_seg: 34.2851, aux.loss_seg: 0.8860, aux.acc_seg: 29.5591, loss: 2.8256
2022-05-22 03:38:31,139 - mmseg - INFO - Iter [4550/320000]	lr: 5.915e-05, eta: 2 days, 3:31:02, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9481, decode.acc_seg: 35.2937, aux.loss_seg: 0.9063, aux.acc_seg: 29.5817, loss: 2.8543
2022-05-22 03:39:00,492 - mmseg - INFO - Iter [4600/320000]	lr: 5.914e-05, eta: 2 days, 3:30:29, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 2.0426, decode.acc_seg: 32.8217, aux.loss_seg: 0.9376, aux.acc_seg: 27.3754, loss: 2.9802
2022-05-22 03:39:29,767 - mmseg - INFO - Iter [4650/320000]	lr: 5.913e-05, eta: 2 days, 3:29:52, time: 0.585, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9512, decode.acc_seg: 35.9977, aux.loss_seg: 0.9026, aux.acc_seg: 30.8100, loss: 2.8539
2022-05-22 03:39:59,032 - mmseg - INFO - Iter [4700/320000]	lr: 5.912e-05, eta: 2 days, 3:29:13, time: 0.585, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0159, decode.acc_seg: 33.9021, aux.loss_seg: 0.9227, aux.acc_seg: 28.9828, loss: 2.9386
2022-05-22 03:40:28,368 - mmseg - INFO - Iter [4750/320000]	lr: 5.911e-05, eta: 2 days, 3:28:40, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0187, decode.acc_seg: 34.3110, aux.loss_seg: 0.9238, aux.acc_seg: 29.1357, loss: 2.9424
2022-05-22 03:40:57,553 - mmseg - INFO - Iter [4800/320000]	lr: 5.910e-05, eta: 2 days, 3:27:57, time: 0.584, data_time: 0.003, memory: 7999, decode.loss_seg: 2.0346, decode.acc_seg: 33.4813, aux.loss_seg: 0.9306, aux.acc_seg: 28.0732, loss: 2.9653
2022-05-22 03:41:27,053 - mmseg - INFO - Iter [4850/320000]	lr: 5.909e-05, eta: 2 days, 3:27:35, time: 0.590, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9541, decode.acc_seg: 34.8815, aux.loss_seg: 0.9026, aux.acc_seg: 29.9004, loss: 2.8567
2022-05-22 03:41:56,397 - mmseg - INFO - Iter [4900/320000]	lr: 5.908e-05, eta: 2 days, 3:27:02, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0482, decode.acc_seg: 36.8811, aux.loss_seg: 0.9360, aux.acc_seg: 32.0990, loss: 2.9842
2022-05-22 03:42:25,706 - mmseg - INFO - Iter [4950/320000]	lr: 5.907e-05, eta: 2 days, 3:26:28, time: 0.586, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0283, decode.acc_seg: 33.7480, aux.loss_seg: 0.9209, aux.acc_seg: 28.7575, loss: 2.9492
2022-05-22 03:42:55,078 - mmseg - INFO - Exp name: bisa_nonorm_0-5.py
2022-05-22 03:42:55,079 - mmseg - INFO - Iter [5000/320000]	lr: 5.906e-05, eta: 2 days, 3:25:57, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9512, decode.acc_seg: 33.1644, aux.loss_seg: 0.8928, aux.acc_seg: 28.2690, loss: 2.8440
2022-05-22 03:43:24,371 - mmseg - INFO - Iter [5050/320000]	lr: 5.905e-05, eta: 2 days, 3:25:22, time: 0.586, data_time: 0.003, memory: 7999, decode.loss_seg: 2.0042, decode.acc_seg: 33.8153, aux.loss_seg: 0.9231, aux.acc_seg: 28.9729, loss: 2.9273
2022-05-22 03:44:08,334 - mmseg - INFO - Iter [5100/320000]	lr: 5.904e-05, eta: 2 days, 3:39:52, time: 0.879, data_time: 0.297, memory: 7999, decode.loss_seg: 1.9817, decode.acc_seg: 36.2698, aux.loss_seg: 0.9008, aux.acc_seg: 31.5548, loss: 2.8825
2022-05-22 03:44:37,576 - mmseg - INFO - Iter [5150/320000]	lr: 5.903e-05, eta: 2 days, 3:39:05, time: 0.585, data_time: 0.003, memory: 7999, decode.loss_seg: 2.0074, decode.acc_seg: 35.3417, aux.loss_seg: 0.9251, aux.acc_seg: 29.5063, loss: 2.9325
2022-05-22 03:45:06,813 - mmseg - INFO - Iter [5200/320000]	lr: 5.903e-05, eta: 2 days, 3:38:18, time: 0.585, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0437, decode.acc_seg: 33.2120, aux.loss_seg: 0.9342, aux.acc_seg: 28.2800, loss: 2.9779
2022-05-22 03:45:36,131 - mmseg - INFO - Iter [5250/320000]	lr: 5.902e-05, eta: 2 days, 3:37:36, time: 0.586, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9475, decode.acc_seg: 33.3413, aux.loss_seg: 0.8964, aux.acc_seg: 28.6399, loss: 2.8439
2022-05-22 03:46:05,330 - mmseg - INFO - Iter [5300/320000]	lr: 5.901e-05, eta: 2 days, 3:36:47, time: 0.584, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9911, decode.acc_seg: 34.2593, aux.loss_seg: 0.9149, aux.acc_seg: 29.0393, loss: 2.9060
2022-05-22 03:46:34,771 - mmseg - INFO - Iter [5350/320000]	lr: 5.900e-05, eta: 2 days, 3:36:13, time: 0.589, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9429, decode.acc_seg: 32.8491, aux.loss_seg: 0.8786, aux.acc_seg: 28.6345, loss: 2.8215
2022-05-22 03:47:04,120 - mmseg - INFO - Iter [5400/320000]	lr: 5.899e-05, eta: 2 days, 3:35:33, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9459, decode.acc_seg: 33.2055, aux.loss_seg: 0.8934, aux.acc_seg: 28.7000, loss: 2.8393
2022-05-22 03:47:33,478 - mmseg - INFO - Iter [5450/320000]	lr: 5.898e-05, eta: 2 days, 3:34:54, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9031, decode.acc_seg: 35.6257, aux.loss_seg: 0.8828, aux.acc_seg: 30.5255, loss: 2.7859
2022-05-22 03:48:02,816 - mmseg - INFO - Iter [5500/320000]	lr: 5.897e-05, eta: 2 days, 3:34:14, time: 0.587, data_time: 0.004, memory: 7999, decode.loss_seg: 1.8029, decode.acc_seg: 36.6456, aux.loss_seg: 0.8474, aux.acc_seg: 30.6118, loss: 2.6503
2022-05-22 03:48:32,135 - mmseg - INFO - Iter [5550/320000]	lr: 5.896e-05, eta: 2 days, 3:33:34, time: 0.586, data_time: 0.004, memory: 7999, decode.loss_seg: 1.9894, decode.acc_seg: 34.8706, aux.loss_seg: 0.9083, aux.acc_seg: 30.0516, loss: 2.8977
2022-05-22 03:49:01,526 - mmseg - INFO - Iter [5600/320000]	lr: 5.895e-05, eta: 2 days, 3:32:57, time: 0.588, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0162, decode.acc_seg: 34.0118, aux.loss_seg: 0.9235, aux.acc_seg: 29.2529, loss: 2.9398
2022-05-22 03:49:30,872 - mmseg - INFO - Iter [5650/320000]	lr: 5.894e-05, eta: 2 days, 3:32:18, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9398, decode.acc_seg: 36.0918, aux.loss_seg: 0.9050, aux.acc_seg: 30.8009, loss: 2.8448
2022-05-22 03:50:00,427 - mmseg - INFO - Iter [5700/320000]	lr: 5.893e-05, eta: 2 days, 3:31:51, time: 0.591, data_time: 0.004, memory: 7999, decode.loss_seg: 1.8996, decode.acc_seg: 34.2129, aux.loss_seg: 0.8837, aux.acc_seg: 28.1557, loss: 2.7833
2022-05-22 03:50:29,783 - mmseg - INFO - Iter [5750/320000]	lr: 5.892e-05, eta: 2 days, 3:31:13, time: 0.587, data_time: 0.003, memory: 7999, decode.loss_seg: 1.9294, decode.acc_seg: 35.7530, aux.loss_seg: 0.8932, aux.acc_seg: 30.2872, loss: 2.8226
2022-05-22 03:50:59,165 - mmseg - INFO - Iter [5800/320000]	lr: 5.891e-05, eta: 2 days, 3:30:37, time: 0.588, data_time: 0.004, memory: 7999, decode.loss_seg: 2.0246, decode.acc_seg: 34.4174, aux.loss_seg: 0.9300, aux.acc_seg: 29.0965, loss: 2.9546
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:58:16,733 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609
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

2022-05-22 03:58:16,733 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:58:17,033 - mmseg - INFO - Config:
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
        bidirectional_lambda_value=0.0,
        lambda_learned=False,
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
work_dir = './work_dirs/bisa_nonorm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:58:19,409 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 03:58:20,069 - mmseg - INFO - Loaded 20210 images
2022-05-22 03:58:22,853 - mmseg - INFO - Loaded 2000 images
2022-05-22 03:58:22,853 - mmseg - INFO - Start running, host: bdevnani3@jill, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_nonorm_0-5
2022-05-22 03:58:22,853 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 03:58:22,853 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 03:58:22,854 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_nonorm_0-5 by HardDiskBackend.
2022-05-22 03:58:35,726 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 03:58:37,702 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
