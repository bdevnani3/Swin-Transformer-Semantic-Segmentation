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
2022-05-22 02:42:24,632 - mmseg - INFO - Environment info:
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
MMSegmentation: 0.11.0+455ebf2
------------------------------------------------------------

2022-05-22 02:42:24,632 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 02:42:24,931 - mmseg - INFO - Config:
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
        reverse_attention_locations=[],
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
work_dir = './work_dirs/bisa_baseline'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 02:42:26,726 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 02:42:27,369 - mmseg - INFO - Loaded 20210 images
2022-05-22 02:42:30,927 - mmseg - INFO - Loaded 2000 images
2022-05-22 02:42:30,927 - mmseg - INFO - Start running, host: bdevnani3@hal, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_baseline
2022-05-22 02:42:30,928 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 02:42:30,928 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 02:42:30,928 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_baseline by HardDiskBackend.
2022-05-22 02:43:01,728 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 02:43:19,627 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 02:46:24,392 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: A40
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

2022-05-22 02:46:24,392 - mmseg - INFO - Distributed training: True
2022-05-22 02:46:24,663 - mmseg - INFO - Config:
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
        reverse_attention_locations=[],
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
work_dir = './work_dirs/bisa_baseline'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 02:46:26,292 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 02:46:28,100 - mmseg - INFO - Loaded 20210 images
2022-05-22 02:46:32,808 - mmseg - INFO - Loaded 2000 images
2022-05-22 02:46:32,809 - mmseg - INFO - Start running, host: bdevnani3@sonny, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_baseline
2022-05-22 02:46:32,809 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 02:46:32,809 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 02:46:32,809 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_baseline by HardDiskBackend.
2022-05-22 02:46:49,730 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 02:46:53,672 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
2022-05-22 02:47:05,870 - mmseg - INFO - Iter [50/320000]	lr: 9.799e-07, eta: 1 day, 12:18:25, time: 0.409, data_time: 0.011, memory: 35360, decode.loss_seg: 4.0467, decode.acc_seg: 0.4610, aux.loss_seg: 1.6049, aux.acc_seg: 0.4820, loss: 5.6517
2022-05-22 02:47:18,376 - mmseg - INFO - Iter [100/320000]	lr: 1.979e-06, eta: 1 day, 5:16:07, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 4.0928, decode.acc_seg: 1.0583, aux.loss_seg: 1.6369, aux.acc_seg: 0.5896, loss: 5.7296
2022-05-22 02:47:30,897 - mmseg - INFO - Iter [150/320000]	lr: 2.979e-06, eta: 1 day, 2:55:33, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 4.0377, decode.acc_seg: 5.1902, aux.loss_seg: 1.6256, aux.acc_seg: 0.9026, loss: 5.6634
2022-05-22 02:47:43,152 - mmseg - INFO - Iter [200/320000]	lr: 3.978e-06, eta: 1 day, 1:37:54, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 3.9278, decode.acc_seg: 8.5040, aux.loss_seg: 1.5973, aux.acc_seg: 1.7896, loss: 5.5250
2022-05-22 02:47:55,345 - mmseg - INFO - Iter [250/320000]	lr: 4.976e-06, eta: 1 day, 0:50:09, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 3.8648, decode.acc_seg: 13.2453, aux.loss_seg: 1.5987, aux.acc_seg: 5.5926, loss: 5.4635
2022-05-22 02:48:07,649 - mmseg - INFO - Iter [300/320000]	lr: 5.974e-06, eta: 1 day, 0:20:01, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 3.8077, decode.acc_seg: 14.3323, aux.loss_seg: 1.5868, aux.acc_seg: 9.2735, loss: 5.3945
2022-05-22 02:48:19,772 - mmseg - INFO - Iter [350/320000]	lr: 6.972e-06, eta: 23:55:50, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 3.8029, decode.acc_seg: 16.2066, aux.loss_seg: 1.6083, aux.acc_seg: 11.6383, loss: 5.4111
2022-05-22 02:48:32,026 - mmseg - INFO - Iter [400/320000]	lr: 7.970e-06, eta: 23:39:20, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 3.6911, decode.acc_seg: 18.2119, aux.loss_seg: 1.5791, aux.acc_seg: 14.3729, loss: 5.2702
2022-05-22 02:48:44,193 - mmseg - INFO - Iter [450/320000]	lr: 8.967e-06, eta: 23:25:26, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 3.6719, decode.acc_seg: 17.3314, aux.loss_seg: 1.5782, aux.acc_seg: 14.2543, loss: 5.2501
2022-05-22 02:48:56,386 - mmseg - INFO - Iter [500/320000]	lr: 9.964e-06, eta: 23:14:29, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 3.5295, decode.acc_seg: 19.0347, aux.loss_seg: 1.5354, aux.acc_seg: 15.6968, loss: 5.0648
2022-05-22 02:49:08,627 - mmseg - INFO - Iter [550/320000]	lr: 1.096e-05, eta: 23:06:05, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 3.5535, decode.acc_seg: 17.5120, aux.loss_seg: 1.5400, aux.acc_seg: 14.1060, loss: 5.0935
2022-05-22 02:49:20,735 - mmseg - INFO - Iter [600/320000]	lr: 1.196e-05, eta: 22:57:44, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 3.3877, decode.acc_seg: 19.6698, aux.loss_seg: 1.4933, aux.acc_seg: 16.5476, loss: 4.8810
2022-05-22 02:49:32,849 - mmseg - INFO - Iter [650/320000]	lr: 1.295e-05, eta: 22:50:47, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 3.4405, decode.acc_seg: 20.2200, aux.loss_seg: 1.5129, aux.acc_seg: 16.6715, loss: 4.9533
2022-05-22 02:49:44,763 - mmseg - INFO - Iter [700/320000]	lr: 1.395e-05, eta: 22:43:15, time: 0.238, data_time: 0.003, memory: 35360, decode.loss_seg: 3.2185, decode.acc_seg: 20.8179, aux.loss_seg: 1.4439, aux.acc_seg: 17.2540, loss: 4.6624
2022-05-22 02:49:56,931 - mmseg - INFO - Iter [750/320000]	lr: 1.494e-05, eta: 22:38:28, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 3.2296, decode.acc_seg: 21.0418, aux.loss_seg: 1.4510, aux.acc_seg: 17.2134, loss: 4.6805
2022-05-22 02:50:08,976 - mmseg - INFO - Iter [800/320000]	lr: 1.594e-05, eta: 22:33:27, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 3.2953, decode.acc_seg: 22.0899, aux.loss_seg: 1.4880, aux.acc_seg: 18.3329, loss: 4.7833
2022-05-22 02:50:20,962 - mmseg - INFO - Iter [850/320000]	lr: 1.693e-05, eta: 22:28:39, time: 0.240, data_time: 0.004, memory: 35360, decode.loss_seg: 3.1536, decode.acc_seg: 21.8159, aux.loss_seg: 1.4328, aux.acc_seg: 18.0123, loss: 4.5864
2022-05-22 02:50:32,979 - mmseg - INFO - Iter [900/320000]	lr: 1.793e-05, eta: 22:24:32, time: 0.240, data_time: 0.003, memory: 35360, decode.loss_seg: 3.0446, decode.acc_seg: 20.3920, aux.loss_seg: 1.3802, aux.acc_seg: 17.0527, loss: 4.4249
2022-05-22 02:50:44,966 - mmseg - INFO - Iter [950/320000]	lr: 1.892e-05, eta: 22:20:41, time: 0.240, data_time: 0.004, memory: 35360, decode.loss_seg: 3.1528, decode.acc_seg: 22.8107, aux.loss_seg: 1.4447, aux.acc_seg: 18.0437, loss: 4.5976
2022-05-22 02:50:57,263 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 02:50:57,263 - mmseg - INFO - Iter [1000/320000]	lr: 1.992e-05, eta: 22:18:49, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 3.0611, decode.acc_seg: 21.4519, aux.loss_seg: 1.3982, aux.acc_seg: 17.2327, loss: 4.4593
2022-05-22 02:51:09,270 - mmseg - INFO - Iter [1050/320000]	lr: 2.091e-05, eta: 22:15:39, time: 0.240, data_time: 0.004, memory: 35360, decode.loss_seg: 2.9668, decode.acc_seg: 22.1081, aux.loss_seg: 1.3650, aux.acc_seg: 17.7362, loss: 4.3318
2022-05-22 02:51:21,311 - mmseg - INFO - Iter [1100/320000]	lr: 2.190e-05, eta: 22:12:54, time: 0.241, data_time: 0.004, memory: 35360, decode.loss_seg: 2.9772, decode.acc_seg: 23.3432, aux.loss_seg: 1.3790, aux.acc_seg: 19.3166, loss: 4.3562
2022-05-22 02:51:33,307 - mmseg - INFO - Iter [1150/320000]	lr: 2.290e-05, eta: 22:10:10, time: 0.240, data_time: 0.003, memory: 35360, decode.loss_seg: 2.9753, decode.acc_seg: 24.5380, aux.loss_seg: 1.3865, aux.acc_seg: 21.0137, loss: 4.3618
2022-05-22 02:51:45,497 - mmseg - INFO - Iter [1200/320000]	lr: 2.389e-05, eta: 22:08:30, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 2.8682, decode.acc_seg: 24.9108, aux.loss_seg: 1.3515, aux.acc_seg: 18.7110, loss: 4.2197
2022-05-22 02:51:57,482 - mmseg - INFO - Iter [1250/320000]	lr: 2.488e-05, eta: 22:06:08, time: 0.240, data_time: 0.003, memory: 35360, decode.loss_seg: 2.8583, decode.acc_seg: 24.2338, aux.loss_seg: 1.3359, aux.acc_seg: 19.3355, loss: 4.1942
2022-05-22 02:52:09,565 - mmseg - INFO - Iter [1300/320000]	lr: 2.587e-05, eta: 22:04:17, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 2.7811, decode.acc_seg: 24.0742, aux.loss_seg: 1.3056, aux.acc_seg: 19.7465, loss: 4.0868
2022-05-22 02:52:21,432 - mmseg - INFO - Iter [1350/320000]	lr: 2.687e-05, eta: 22:01:44, time: 0.237, data_time: 0.003, memory: 35360, decode.loss_seg: 2.6865, decode.acc_seg: 23.4247, aux.loss_seg: 1.2679, aux.acc_seg: 19.2850, loss: 3.9544
2022-05-22 02:52:33,507 - mmseg - INFO - Iter [1400/320000]	lr: 2.786e-05, eta: 22:00:07, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 2.6393, decode.acc_seg: 24.1312, aux.loss_seg: 1.2483, aux.acc_seg: 19.6920, loss: 3.8877
2022-05-22 02:52:45,440 - mmseg - INFO - Iter [1450/320000]	lr: 2.885e-05, eta: 21:58:06, time: 0.239, data_time: 0.004, memory: 35360, decode.loss_seg: 2.6872, decode.acc_seg: 23.0907, aux.loss_seg: 1.2617, aux.acc_seg: 18.6910, loss: 3.9488
2022-05-22 02:52:57,445 - mmseg - INFO - Iter [1500/320000]	lr: 2.984e-05, eta: 21:56:26, time: 0.240, data_time: 0.003, memory: 35360, decode.loss_seg: 2.5803, decode.acc_seg: 25.2802, aux.loss_seg: 1.2244, aux.acc_seg: 21.6013, loss: 3.8047
2022-05-22 02:53:09,359 - mmseg - INFO - Iter [1550/320000]	lr: 3.083e-05, eta: 21:54:34, time: 0.238, data_time: 0.003, memory: 35360, decode.loss_seg: 2.6078, decode.acc_seg: 25.3082, aux.loss_seg: 1.2336, aux.acc_seg: 20.8805, loss: 3.8414
2022-05-22 02:53:21,264 - mmseg - INFO - Iter [1600/320000]	lr: 3.182e-05, eta: 21:52:46, time: 0.238, data_time: 0.003, memory: 35360, decode.loss_seg: 2.4918, decode.acc_seg: 25.8468, aux.loss_seg: 1.1790, aux.acc_seg: 19.9728, loss: 3.6708
2022-05-22 02:53:33,299 - mmseg - INFO - Iter [1650/320000]	lr: 3.281e-05, eta: 21:51:29, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 2.6986, decode.acc_seg: 24.6556, aux.loss_seg: 1.2490, aux.acc_seg: 20.0571, loss: 3.9476
2022-05-22 02:53:45,348 - mmseg - INFO - Iter [1700/320000]	lr: 3.380e-05, eta: 21:50:19, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 2.4696, decode.acc_seg: 27.7272, aux.loss_seg: 1.1711, aux.acc_seg: 21.2101, loss: 3.6407
2022-05-22 02:53:57,421 - mmseg - INFO - Iter [1750/320000]	lr: 3.479e-05, eta: 21:49:16, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 2.3359, decode.acc_seg: 25.0506, aux.loss_seg: 1.1006, aux.acc_seg: 20.3775, loss: 3.4364
2022-05-22 02:54:09,467 - mmseg - INFO - Iter [1800/320000]	lr: 3.578e-05, eta: 21:48:11, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 2.4750, decode.acc_seg: 26.3270, aux.loss_seg: 1.1619, aux.acc_seg: 21.2382, loss: 3.6369
2022-05-22 02:54:21,398 - mmseg - INFO - Iter [1850/320000]	lr: 3.677e-05, eta: 21:46:49, time: 0.239, data_time: 0.003, memory: 35360, decode.loss_seg: 2.5132, decode.acc_seg: 25.9384, aux.loss_seg: 1.1738, aux.acc_seg: 21.4343, loss: 3.6870
2022-05-22 02:54:33,536 - mmseg - INFO - Iter [1900/320000]	lr: 3.775e-05, eta: 21:46:06, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 2.4262, decode.acc_seg: 25.4996, aux.loss_seg: 1.1216, aux.acc_seg: 20.0209, loss: 3.5478
2022-05-22 02:54:45,452 - mmseg - INFO - Iter [1950/320000]	lr: 3.874e-05, eta: 21:44:48, time: 0.238, data_time: 0.003, memory: 35360, decode.loss_seg: 2.4324, decode.acc_seg: 27.3563, aux.loss_seg: 1.1418, aux.acc_seg: 22.3053, loss: 3.5742
2022-05-22 02:54:57,594 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 02:54:57,594 - mmseg - INFO - Iter [2000/320000]	lr: 3.973e-05, eta: 21:44:09, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 2.5359, decode.acc_seg: 26.8719, aux.loss_seg: 1.1640, aux.acc_seg: 22.2894, loss: 3.6999
2022-05-22 02:55:09,526 - mmseg - INFO - Iter [2050/320000]	lr: 4.072e-05, eta: 21:42:59, time: 0.239, data_time: 0.003, memory: 35360, decode.loss_seg: 2.4096, decode.acc_seg: 27.9181, aux.loss_seg: 1.1154, aux.acc_seg: 22.4882, loss: 3.5251
2022-05-22 02:55:21,584 - mmseg - INFO - Iter [2100/320000]	lr: 4.170e-05, eta: 21:42:11, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 2.5751, decode.acc_seg: 26.3713, aux.loss_seg: 1.1668, aux.acc_seg: 22.3548, loss: 3.7419
2022-05-22 02:55:33,614 - mmseg - INFO - Iter [2150/320000]	lr: 4.269e-05, eta: 21:41:21, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 2.3767, decode.acc_seg: 27.5385, aux.loss_seg: 1.0917, aux.acc_seg: 23.5240, loss: 3.4684
2022-05-22 02:55:45,841 - mmseg - INFO - Iter [2200/320000]	lr: 4.368e-05, eta: 21:41:00, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 2.4047, decode.acc_seg: 26.2621, aux.loss_seg: 1.0936, aux.acc_seg: 21.7050, loss: 3.4983
2022-05-22 02:55:57,866 - mmseg - INFO - Iter [2250/320000]	lr: 4.466e-05, eta: 21:40:11, time: 0.240, data_time: 0.003, memory: 35360, decode.loss_seg: 2.3409, decode.acc_seg: 27.9171, aux.loss_seg: 1.0733, aux.acc_seg: 22.2073, loss: 3.4142
2022-05-22 02:56:09,975 - mmseg - INFO - Iter [2300/320000]	lr: 4.565e-05, eta: 21:39:36, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 2.2133, decode.acc_seg: 27.8651, aux.loss_seg: 1.0219, aux.acc_seg: 23.8677, loss: 3.2352
2022-05-22 02:56:22,264 - mmseg - INFO - Iter [2350/320000]	lr: 4.664e-05, eta: 21:39:27, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 2.3306, decode.acc_seg: 29.3450, aux.loss_seg: 1.0579, aux.acc_seg: 24.5776, loss: 3.3885
2022-05-22 02:56:34,449 - mmseg - INFO - Iter [2400/320000]	lr: 4.762e-05, eta: 21:39:02, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 2.4153, decode.acc_seg: 27.2113, aux.loss_seg: 1.0951, aux.acc_seg: 22.5118, loss: 3.5104
2022-05-22 02:56:46,892 - mmseg - INFO - Iter [2450/320000]	lr: 4.861e-05, eta: 21:39:12, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 2.2797, decode.acc_seg: 29.7670, aux.loss_seg: 1.0433, aux.acc_seg: 24.6386, loss: 3.3230
2022-05-22 02:56:59,146 - mmseg - INFO - Iter [2500/320000]	lr: 4.959e-05, eta: 21:38:57, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 2.3080, decode.acc_seg: 28.4177, aux.loss_seg: 1.0453, aux.acc_seg: 23.9157, loss: 3.3533
2022-05-22 02:57:28,200 - mmseg - INFO - Iter [2550/320000]	lr: 5.057e-05, eta: 22:13:34, time: 0.581, data_time: 0.343, memory: 35360, decode.loss_seg: 2.2702, decode.acc_seg: 30.0473, aux.loss_seg: 1.0388, aux.acc_seg: 25.6924, loss: 3.3091
2022-05-22 02:57:40,532 - mmseg - INFO - Iter [2600/320000]	lr: 5.156e-05, eta: 22:12:48, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 2.2776, decode.acc_seg: 29.4650, aux.loss_seg: 1.0334, aux.acc_seg: 25.2751, loss: 3.3111
2022-05-22 02:57:52,813 - mmseg - INFO - Iter [2650/320000]	lr: 5.254e-05, eta: 22:11:58, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 2.2458, decode.acc_seg: 28.6287, aux.loss_seg: 1.0179, aux.acc_seg: 24.0397, loss: 3.2636
2022-05-22 02:58:05,050 - mmseg - INFO - Iter [2700/320000]	lr: 5.352e-05, eta: 22:11:03, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 2.2431, decode.acc_seg: 29.5204, aux.loss_seg: 1.0167, aux.acc_seg: 24.7535, loss: 3.2598
2022-05-22 02:58:17,274 - mmseg - INFO - Iter [2750/320000]	lr: 5.451e-05, eta: 22:10:09, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 2.3084, decode.acc_seg: 28.3335, aux.loss_seg: 1.0402, aux.acc_seg: 24.1226, loss: 3.3485
2022-05-22 02:58:29,576 - mmseg - INFO - Iter [2800/320000]	lr: 5.549e-05, eta: 22:09:25, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 2.3071, decode.acc_seg: 29.4726, aux.loss_seg: 1.0400, aux.acc_seg: 25.4014, loss: 3.3471
2022-05-22 02:58:41,918 - mmseg - INFO - Iter [2850/320000]	lr: 5.647e-05, eta: 22:08:47, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 2.2809, decode.acc_seg: 29.3410, aux.loss_seg: 1.0279, aux.acc_seg: 24.9067, loss: 3.3088
2022-05-22 02:58:54,216 - mmseg - INFO - Iter [2900/320000]	lr: 5.745e-05, eta: 22:08:05, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 2.2649, decode.acc_seg: 27.7553, aux.loss_seg: 1.0171, aux.acc_seg: 24.1418, loss: 3.2821
2022-05-22 02:59:06,377 - mmseg - INFO - Iter [2950/320000]	lr: 5.844e-05, eta: 22:07:09, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 2.2827, decode.acc_seg: 31.1413, aux.loss_seg: 1.0336, aux.acc_seg: 26.5049, loss: 3.3164
2022-05-22 02:59:18,665 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 02:59:18,665 - mmseg - INFO - Iter [3000/320000]	lr: 5.942e-05, eta: 22:06:27, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1354, decode.acc_seg: 31.7249, aux.loss_seg: 0.9772, aux.acc_seg: 27.5760, loss: 3.1126
2022-05-22 02:59:30,842 - mmseg - INFO - Iter [3050/320000]	lr: 5.943e-05, eta: 22:05:36, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 2.2127, decode.acc_seg: 31.5398, aux.loss_seg: 0.9986, aux.acc_seg: 27.0105, loss: 3.2113
2022-05-22 02:59:43,027 - mmseg - INFO - Iter [3100/320000]	lr: 5.942e-05, eta: 22:04:46, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1223, decode.acc_seg: 30.6028, aux.loss_seg: 0.9657, aux.acc_seg: 25.7539, loss: 3.0879
2022-05-22 02:59:55,381 - mmseg - INFO - Iter [3150/320000]	lr: 5.941e-05, eta: 22:04:14, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1669, decode.acc_seg: 29.1038, aux.loss_seg: 0.9778, aux.acc_seg: 25.6570, loss: 3.1446
2022-05-22 03:00:07,633 - mmseg - INFO - Iter [3200/320000]	lr: 5.940e-05, eta: 22:03:34, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 2.2240, decode.acc_seg: 30.9439, aux.loss_seg: 1.0013, aux.acc_seg: 28.0877, loss: 3.2253
2022-05-22 03:00:19,951 - mmseg - INFO - Iter [3250/320000]	lr: 5.939e-05, eta: 22:03:00, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1854, decode.acc_seg: 30.5693, aux.loss_seg: 0.9908, aux.acc_seg: 26.3360, loss: 3.1762
2022-05-22 03:00:32,282 - mmseg - INFO - Iter [3300/320000]	lr: 5.938e-05, eta: 22:02:29, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 2.1988, decode.acc_seg: 29.4622, aux.loss_seg: 0.9950, aux.acc_seg: 25.1474, loss: 3.1939
2022-05-22 03:00:44,529 - mmseg - INFO - Iter [3350/320000]	lr: 5.937e-05, eta: 22:01:50, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 2.1886, decode.acc_seg: 31.6016, aux.loss_seg: 0.9974, aux.acc_seg: 27.1667, loss: 3.1860
2022-05-22 03:00:56,874 - mmseg - INFO - Iter [3400/320000]	lr: 5.936e-05, eta: 22:01:20, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1848, decode.acc_seg: 31.6992, aux.loss_seg: 0.9914, aux.acc_seg: 27.2392, loss: 3.1761
2022-05-22 03:01:09,227 - mmseg - INFO - Iter [3450/320000]	lr: 5.935e-05, eta: 22:00:52, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 2.2579, decode.acc_seg: 31.3671, aux.loss_seg: 1.0236, aux.acc_seg: 27.0303, loss: 3.2815
2022-05-22 03:01:21,864 - mmseg - INFO - Iter [3500/320000]	lr: 5.934e-05, eta: 22:00:50, time: 0.253, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0408, decode.acc_seg: 33.3153, aux.loss_seg: 0.9337, aux.acc_seg: 27.6976, loss: 2.9746
2022-05-22 03:01:34,210 - mmseg - INFO - Iter [3550/320000]	lr: 5.933e-05, eta: 22:00:22, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 2.1338, decode.acc_seg: 32.1153, aux.loss_seg: 0.9731, aux.acc_seg: 27.8620, loss: 3.1069
2022-05-22 03:01:46,516 - mmseg - INFO - Iter [3600/320000]	lr: 5.933e-05, eta: 21:59:51, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1750, decode.acc_seg: 30.7813, aux.loss_seg: 0.9849, aux.acc_seg: 26.5801, loss: 3.1599
2022-05-22 03:01:58,839 - mmseg - INFO - Iter [3650/320000]	lr: 5.932e-05, eta: 21:59:22, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0852, decode.acc_seg: 34.4150, aux.loss_seg: 0.9617, aux.acc_seg: 30.0989, loss: 3.0469
2022-05-22 03:02:11,290 - mmseg - INFO - Iter [3700/320000]	lr: 5.931e-05, eta: 21:59:04, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0779, decode.acc_seg: 30.7934, aux.loss_seg: 0.9508, aux.acc_seg: 25.3750, loss: 3.0288
2022-05-22 03:02:23,593 - mmseg - INFO - Iter [3750/320000]	lr: 5.930e-05, eta: 21:58:34, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0771, decode.acc_seg: 31.3326, aux.loss_seg: 0.9435, aux.acc_seg: 26.1914, loss: 3.0206
2022-05-22 03:02:35,931 - mmseg - INFO - Iter [3800/320000]	lr: 5.929e-05, eta: 21:58:08, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0523, decode.acc_seg: 30.8007, aux.loss_seg: 0.9337, aux.acc_seg: 26.7317, loss: 2.9860
2022-05-22 03:02:48,257 - mmseg - INFO - Iter [3850/320000]	lr: 5.928e-05, eta: 21:57:40, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1432, decode.acc_seg: 31.8276, aux.loss_seg: 0.9785, aux.acc_seg: 26.9222, loss: 3.1217
2022-05-22 03:03:00,478 - mmseg - INFO - Iter [3900/320000]	lr: 5.927e-05, eta: 21:57:05, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 2.1088, decode.acc_seg: 32.7786, aux.loss_seg: 0.9645, aux.acc_seg: 27.8704, loss: 3.0733
2022-05-22 03:03:12,913 - mmseg - INFO - Iter [3950/320000]	lr: 5.926e-05, eta: 21:56:47, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9762, decode.acc_seg: 32.4024, aux.loss_seg: 0.9001, aux.acc_seg: 27.6490, loss: 2.8763
2022-05-22 03:03:25,172 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:03:25,172 - mmseg - INFO - Iter [4000/320000]	lr: 5.925e-05, eta: 21:56:15, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9718, decode.acc_seg: 33.7626, aux.loss_seg: 0.9113, aux.acc_seg: 28.7374, loss: 2.8831
2022-05-22 03:03:37,501 - mmseg - INFO - Iter [4050/320000]	lr: 5.924e-05, eta: 21:55:50, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0484, decode.acc_seg: 34.3217, aux.loss_seg: 0.9349, aux.acc_seg: 29.1538, loss: 2.9832
2022-05-22 03:03:49,684 - mmseg - INFO - Iter [4100/320000]	lr: 5.923e-05, eta: 21:55:14, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 2.1420, decode.acc_seg: 33.4985, aux.loss_seg: 0.9657, aux.acc_seg: 27.9351, loss: 3.1077
2022-05-22 03:04:02,013 - mmseg - INFO - Iter [4150/320000]	lr: 5.922e-05, eta: 21:54:49, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0047, decode.acc_seg: 34.8491, aux.loss_seg: 0.9215, aux.acc_seg: 29.7259, loss: 2.9263
2022-05-22 03:04:14,284 - mmseg - INFO - Iter [4200/320000]	lr: 5.921e-05, eta: 21:54:20, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9717, decode.acc_seg: 34.1311, aux.loss_seg: 0.9097, aux.acc_seg: 29.1498, loss: 2.8814
2022-05-22 03:04:26,548 - mmseg - INFO - Iter [4250/320000]	lr: 5.920e-05, eta: 21:53:51, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9778, decode.acc_seg: 34.3809, aux.loss_seg: 0.9142, aux.acc_seg: 29.3665, loss: 2.8920
2022-05-22 03:04:38,787 - mmseg - INFO - Iter [4300/320000]	lr: 5.919e-05, eta: 21:53:20, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9429, decode.acc_seg: 35.7568, aux.loss_seg: 0.9025, aux.acc_seg: 30.0678, loss: 2.8454
2022-05-22 03:04:51,032 - mmseg - INFO - Iter [4350/320000]	lr: 5.918e-05, eta: 21:52:51, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 2.0348, decode.acc_seg: 33.6933, aux.loss_seg: 0.9293, aux.acc_seg: 29.0907, loss: 2.9641
2022-05-22 03:05:03,278 - mmseg - INFO - Iter [4400/320000]	lr: 5.918e-05, eta: 21:52:22, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9809, decode.acc_seg: 33.3425, aux.loss_seg: 0.9056, aux.acc_seg: 28.8010, loss: 2.8865
2022-05-22 03:05:15,599 - mmseg - INFO - Iter [4450/320000]	lr: 5.917e-05, eta: 21:51:58, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9859, decode.acc_seg: 35.2384, aux.loss_seg: 0.9163, aux.acc_seg: 29.7245, loss: 2.9023
2022-05-22 03:05:27,809 - mmseg - INFO - Iter [4500/320000]	lr: 5.916e-05, eta: 21:51:27, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9406, decode.acc_seg: 34.2360, aux.loss_seg: 0.8834, aux.acc_seg: 30.1146, loss: 2.8239
2022-05-22 03:05:40,130 - mmseg - INFO - Iter [4550/320000]	lr: 5.915e-05, eta: 21:51:04, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9735, decode.acc_seg: 34.3999, aux.loss_seg: 0.9155, aux.acc_seg: 28.1986, loss: 2.8891
2022-05-22 03:05:52,431 - mmseg - INFO - Iter [4600/320000]	lr: 5.914e-05, eta: 21:50:40, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9858, decode.acc_seg: 32.7971, aux.loss_seg: 0.9139, aux.acc_seg: 27.2590, loss: 2.8996
2022-05-22 03:06:04,597 - mmseg - INFO - Iter [4650/320000]	lr: 5.913e-05, eta: 21:50:07, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9127, decode.acc_seg: 35.8660, aux.loss_seg: 0.8954, aux.acc_seg: 30.0955, loss: 2.8081
2022-05-22 03:06:16,902 - mmseg - INFO - Iter [4700/320000]	lr: 5.912e-05, eta: 21:49:44, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9932, decode.acc_seg: 33.0384, aux.loss_seg: 0.9156, aux.acc_seg: 28.4230, loss: 2.9088
2022-05-22 03:06:29,114 - mmseg - INFO - Iter [4750/320000]	lr: 5.911e-05, eta: 21:49:15, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0061, decode.acc_seg: 33.4656, aux.loss_seg: 0.9165, aux.acc_seg: 28.7097, loss: 2.9226
2022-05-22 03:06:41,535 - mmseg - INFO - Iter [4800/320000]	lr: 5.910e-05, eta: 21:49:00, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9704, decode.acc_seg: 33.3683, aux.loss_seg: 0.9004, aux.acc_seg: 28.4721, loss: 2.8708
2022-05-22 03:06:53,847 - mmseg - INFO - Iter [4850/320000]	lr: 5.909e-05, eta: 21:48:38, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 2.0268, decode.acc_seg: 34.1064, aux.loss_seg: 0.9289, aux.acc_seg: 28.8255, loss: 2.9557
2022-05-22 03:07:06,085 - mmseg - INFO - Iter [4900/320000]	lr: 5.908e-05, eta: 21:48:12, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 2.0016, decode.acc_seg: 36.2491, aux.loss_seg: 0.9207, aux.acc_seg: 31.0775, loss: 2.9223
2022-05-22 03:07:18,794 - mmseg - INFO - Iter [4950/320000]	lr: 5.907e-05, eta: 21:48:15, time: 0.254, data_time: 0.003, memory: 35360, decode.loss_seg: 2.0657, decode.acc_seg: 34.7136, aux.loss_seg: 0.9457, aux.acc_seg: 29.7196, loss: 3.0114
2022-05-22 03:07:31,257 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:07:31,257 - mmseg - INFO - Iter [5000/320000]	lr: 5.906e-05, eta: 21:48:03, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9724, decode.acc_seg: 33.0709, aux.loss_seg: 0.9069, aux.acc_seg: 27.9145, loss: 2.8793
2022-05-22 03:07:43,448 - mmseg - INFO - Iter [5050/320000]	lr: 5.905e-05, eta: 21:47:34, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9707, decode.acc_seg: 33.7997, aux.loss_seg: 0.9060, aux.acc_seg: 28.9740, loss: 2.8767
2022-05-22 03:08:12,037 - mmseg - INFO - Iter [5100/320000]	lr: 5.904e-05, eta: 22:03:58, time: 0.572, data_time: 0.327, memory: 35360, decode.loss_seg: 1.8847, decode.acc_seg: 36.9839, aux.loss_seg: 0.8693, aux.acc_seg: 32.0016, loss: 2.7540
2022-05-22 03:08:24,634 - mmseg - INFO - Iter [5150/320000]	lr: 5.903e-05, eta: 22:03:44, time: 0.252, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9122, decode.acc_seg: 36.3136, aux.loss_seg: 0.8934, aux.acc_seg: 30.4683, loss: 2.8056
2022-05-22 03:08:37,019 - mmseg - INFO - Iter [5200/320000]	lr: 5.903e-05, eta: 22:03:18, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9823, decode.acc_seg: 35.3293, aux.loss_seg: 0.9211, aux.acc_seg: 29.7860, loss: 2.9034
2022-05-22 03:08:49,592 - mmseg - INFO - Iter [5250/320000]	lr: 5.902e-05, eta: 22:03:03, time: 0.251, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9777, decode.acc_seg: 32.5450, aux.loss_seg: 0.9121, aux.acc_seg: 27.6472, loss: 2.8898
2022-05-22 03:09:02,095 - mmseg - INFO - Iter [5300/320000]	lr: 5.901e-05, eta: 22:02:44, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8868, decode.acc_seg: 34.4547, aux.loss_seg: 0.8821, aux.acc_seg: 29.1824, loss: 2.7689
2022-05-22 03:09:14,457 - mmseg - INFO - Iter [5350/320000]	lr: 5.900e-05, eta: 22:02:16, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9483, decode.acc_seg: 33.6810, aux.loss_seg: 0.8875, aux.acc_seg: 28.9529, loss: 2.8358
2022-05-22 03:09:26,844 - mmseg - INFO - Iter [5400/320000]	lr: 5.899e-05, eta: 22:01:51, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9592, decode.acc_seg: 33.7372, aux.loss_seg: 0.8961, aux.acc_seg: 29.3649, loss: 2.8553
2022-05-22 03:09:39,178 - mmseg - INFO - Iter [5450/320000]	lr: 5.898e-05, eta: 22:01:23, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 2.0201, decode.acc_seg: 34.7413, aux.loss_seg: 0.9247, aux.acc_seg: 30.4142, loss: 2.9448
2022-05-22 03:09:51,561 - mmseg - INFO - Iter [5500/320000]	lr: 5.897e-05, eta: 22:00:57, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7445, decode.acc_seg: 37.5817, aux.loss_seg: 0.8243, aux.acc_seg: 31.3969, loss: 2.5689
2022-05-22 03:10:04,269 - mmseg - INFO - Iter [5550/320000]	lr: 5.896e-05, eta: 22:00:51, time: 0.254, data_time: 0.005, memory: 35360, decode.loss_seg: 1.9887, decode.acc_seg: 35.0540, aux.loss_seg: 0.9208, aux.acc_seg: 30.1964, loss: 2.9096
2022-05-22 03:10:16,848 - mmseg - INFO - Iter [5600/320000]	lr: 5.895e-05, eta: 22:00:37, time: 0.252, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9380, decode.acc_seg: 34.0451, aux.loss_seg: 0.8990, aux.acc_seg: 29.0892, loss: 2.8370
2022-05-22 03:10:29,304 - mmseg - INFO - Iter [5650/320000]	lr: 5.894e-05, eta: 22:00:16, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8690, decode.acc_seg: 37.0035, aux.loss_seg: 0.8783, aux.acc_seg: 31.1017, loss: 2.7474
2022-05-22 03:10:41,601 - mmseg - INFO - Iter [5700/320000]	lr: 5.893e-05, eta: 21:59:47, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8739, decode.acc_seg: 34.5143, aux.loss_seg: 0.8754, aux.acc_seg: 29.1199, loss: 2.7493
2022-05-22 03:10:54,356 - mmseg - INFO - Iter [5750/320000]	lr: 5.892e-05, eta: 21:59:43, time: 0.255, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9308, decode.acc_seg: 34.3836, aux.loss_seg: 0.8933, aux.acc_seg: 29.2056, loss: 2.8240
2022-05-22 03:11:06,610 - mmseg - INFO - Iter [5800/320000]	lr: 5.891e-05, eta: 21:59:12, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8753, decode.acc_seg: 35.3965, aux.loss_seg: 0.8773, aux.acc_seg: 29.6643, loss: 2.7527
2022-05-22 03:11:18,762 - mmseg - INFO - Iter [5850/320000]	lr: 5.890e-05, eta: 21:58:35, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9702, decode.acc_seg: 33.8519, aux.loss_seg: 0.9036, aux.acc_seg: 28.5668, loss: 2.8739
2022-05-22 03:11:30,902 - mmseg - INFO - Iter [5900/320000]	lr: 5.889e-05, eta: 21:57:59, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7957, decode.acc_seg: 35.6711, aux.loss_seg: 0.8319, aux.acc_seg: 30.1658, loss: 2.6276
2022-05-22 03:11:43,257 - mmseg - INFO - Iter [5950/320000]	lr: 5.888e-05, eta: 21:57:34, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9090, decode.acc_seg: 32.4990, aux.loss_seg: 0.8796, aux.acc_seg: 27.3417, loss: 2.7886
2022-05-22 03:11:55,518 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:11:55,518 - mmseg - INFO - Iter [6000/320000]	lr: 5.888e-05, eta: 21:57:04, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9323, decode.acc_seg: 35.0653, aux.loss_seg: 0.8826, aux.acc_seg: 30.0052, loss: 2.8149
2022-05-22 03:12:07,895 - mmseg - INFO - Iter [6050/320000]	lr: 5.887e-05, eta: 21:56:41, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9126, decode.acc_seg: 33.0817, aux.loss_seg: 0.8817, aux.acc_seg: 28.5475, loss: 2.7943
2022-05-22 03:12:20,511 - mmseg - INFO - Iter [6100/320000]	lr: 5.886e-05, eta: 21:56:30, time: 0.252, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8976, decode.acc_seg: 36.8202, aux.loss_seg: 0.8808, aux.acc_seg: 31.5406, loss: 2.7784
2022-05-22 03:12:32,826 - mmseg - INFO - Iter [6150/320000]	lr: 5.885e-05, eta: 21:56:04, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9126, decode.acc_seg: 36.3878, aux.loss_seg: 0.8913, aux.acc_seg: 30.7073, loss: 2.8039
2022-05-22 03:12:45,188 - mmseg - INFO - Iter [6200/320000]	lr: 5.884e-05, eta: 21:55:40, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8006, decode.acc_seg: 37.8346, aux.loss_seg: 0.8452, aux.acc_seg: 31.5972, loss: 2.6458
2022-05-22 03:12:57,424 - mmseg - INFO - Iter [6250/320000]	lr: 5.883e-05, eta: 21:55:10, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8747, decode.acc_seg: 35.2012, aux.loss_seg: 0.8710, aux.acc_seg: 30.0041, loss: 2.7457
2022-05-22 03:13:09,848 - mmseg - INFO - Iter [6300/320000]	lr: 5.882e-05, eta: 21:54:50, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8542, decode.acc_seg: 37.1873, aux.loss_seg: 0.8633, aux.acc_seg: 31.7243, loss: 2.7175
2022-05-22 03:13:22,231 - mmseg - INFO - Iter [6350/320000]	lr: 5.881e-05, eta: 21:54:28, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8337, decode.acc_seg: 37.6483, aux.loss_seg: 0.8751, aux.acc_seg: 31.3898, loss: 2.7087
2022-05-22 03:13:34,490 - mmseg - INFO - Iter [6400/320000]	lr: 5.880e-05, eta: 21:54:00, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8734, decode.acc_seg: 36.3399, aux.loss_seg: 0.8690, aux.acc_seg: 31.9502, loss: 2.7424
2022-05-22 03:13:46,794 - mmseg - INFO - Iter [6450/320000]	lr: 5.879e-05, eta: 21:53:34, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8713, decode.acc_seg: 34.6243, aux.loss_seg: 0.8610, aux.acc_seg: 29.5134, loss: 2.7323
2022-05-22 03:13:59,244 - mmseg - INFO - Iter [6500/320000]	lr: 5.878e-05, eta: 21:53:16, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7920, decode.acc_seg: 36.1426, aux.loss_seg: 0.8421, aux.acc_seg: 31.3327, loss: 2.6341
2022-05-22 03:14:11,464 - mmseg - INFO - Iter [6550/320000]	lr: 5.877e-05, eta: 21:52:47, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9011, decode.acc_seg: 36.7046, aux.loss_seg: 0.8914, aux.acc_seg: 30.0848, loss: 2.7925
2022-05-22 03:14:23,698 - mmseg - INFO - Iter [6600/320000]	lr: 5.876e-05, eta: 21:52:19, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7931, decode.acc_seg: 38.7939, aux.loss_seg: 0.8440, aux.acc_seg: 33.2795, loss: 2.6372
2022-05-22 03:14:35,994 - mmseg - INFO - Iter [6650/320000]	lr: 5.875e-05, eta: 21:51:54, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9518, decode.acc_seg: 35.1861, aux.loss_seg: 0.8967, aux.acc_seg: 30.2186, loss: 2.8485
2022-05-22 03:14:48,426 - mmseg - INFO - Iter [6700/320000]	lr: 5.874e-05, eta: 21:51:35, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9387, decode.acc_seg: 35.6842, aux.loss_seg: 0.8867, aux.acc_seg: 30.8528, loss: 2.8254
2022-05-22 03:15:00,803 - mmseg - INFO - Iter [6750/320000]	lr: 5.873e-05, eta: 21:51:14, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9063, decode.acc_seg: 34.8662, aux.loss_seg: 0.8810, aux.acc_seg: 30.7358, loss: 2.7873
2022-05-22 03:15:13,238 - mmseg - INFO - Iter [6800/320000]	lr: 5.873e-05, eta: 21:50:56, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8590, decode.acc_seg: 35.3002, aux.loss_seg: 0.8699, aux.acc_seg: 29.4879, loss: 2.7288
2022-05-22 03:15:25,466 - mmseg - INFO - Iter [6850/320000]	lr: 5.872e-05, eta: 21:50:28, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8307, decode.acc_seg: 38.8736, aux.loss_seg: 0.8596, aux.acc_seg: 32.3707, loss: 2.6903
2022-05-22 03:15:38,038 - mmseg - INFO - Iter [6900/320000]	lr: 5.871e-05, eta: 21:50:16, time: 0.251, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8671, decode.acc_seg: 37.5072, aux.loss_seg: 0.8741, aux.acc_seg: 32.8434, loss: 2.7411
2022-05-22 03:15:50,263 - mmseg - INFO - Iter [6950/320000]	lr: 5.870e-05, eta: 21:49:49, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8523, decode.acc_seg: 35.6320, aux.loss_seg: 0.8659, aux.acc_seg: 30.8640, loss: 2.7182
2022-05-22 03:16:02,560 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:16:02,561 - mmseg - INFO - Iter [7000/320000]	lr: 5.869e-05, eta: 21:49:25, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9167, decode.acc_seg: 35.8409, aux.loss_seg: 0.8894, aux.acc_seg: 30.4891, loss: 2.8060
2022-05-22 03:16:14,972 - mmseg - INFO - Iter [7050/320000]	lr: 5.868e-05, eta: 21:49:06, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8327, decode.acc_seg: 36.3140, aux.loss_seg: 0.8539, aux.acc_seg: 30.7455, loss: 2.6865
2022-05-22 03:16:27,213 - mmseg - INFO - Iter [7100/320000]	lr: 5.867e-05, eta: 21:48:40, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7225, decode.acc_seg: 39.0401, aux.loss_seg: 0.8109, aux.acc_seg: 33.2723, loss: 2.5334
2022-05-22 03:16:39,942 - mmseg - INFO - Iter [7150/320000]	lr: 5.866e-05, eta: 21:48:36, time: 0.255, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9259, decode.acc_seg: 35.6514, aux.loss_seg: 0.8841, aux.acc_seg: 30.7258, loss: 2.8100
2022-05-22 03:16:52,191 - mmseg - INFO - Iter [7200/320000]	lr: 5.865e-05, eta: 21:48:10, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8175, decode.acc_seg: 38.3828, aux.loss_seg: 0.8465, aux.acc_seg: 32.3529, loss: 2.6640
2022-05-22 03:17:04,384 - mmseg - INFO - Iter [7250/320000]	lr: 5.864e-05, eta: 21:47:42, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7932, decode.acc_seg: 38.3453, aux.loss_seg: 0.8467, aux.acc_seg: 32.9426, loss: 2.6399
2022-05-22 03:17:16,779 - mmseg - INFO - Iter [7300/320000]	lr: 5.863e-05, eta: 21:47:23, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9052, decode.acc_seg: 35.9105, aux.loss_seg: 0.8757, aux.acc_seg: 30.4633, loss: 2.7809
2022-05-22 03:17:28,998 - mmseg - INFO - Iter [7350/320000]	lr: 5.862e-05, eta: 21:46:57, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8487, decode.acc_seg: 37.5303, aux.loss_seg: 0.8606, aux.acc_seg: 32.2959, loss: 2.7092
2022-05-22 03:17:41,314 - mmseg - INFO - Iter [7400/320000]	lr: 5.861e-05, eta: 21:46:35, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8234, decode.acc_seg: 35.0829, aux.loss_seg: 0.8458, aux.acc_seg: 29.7859, loss: 2.6692
2022-05-22 03:17:53,493 - mmseg - INFO - Iter [7450/320000]	lr: 5.860e-05, eta: 21:46:07, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8261, decode.acc_seg: 36.4658, aux.loss_seg: 0.8556, aux.acc_seg: 30.8364, loss: 2.6816
2022-05-22 03:18:05,617 - mmseg - INFO - Iter [7500/320000]	lr: 5.859e-05, eta: 21:45:37, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8470, decode.acc_seg: 38.0997, aux.loss_seg: 0.8625, aux.acc_seg: 32.2600, loss: 2.7095
2022-05-22 03:18:17,842 - mmseg - INFO - Iter [7550/320000]	lr: 5.858e-05, eta: 21:45:12, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8040, decode.acc_seg: 39.3169, aux.loss_seg: 0.8474, aux.acc_seg: 34.3999, loss: 2.6513
2022-05-22 03:18:46,641 - mmseg - INFO - Iter [7600/320000]	lr: 5.858e-05, eta: 21:56:08, time: 0.576, data_time: 0.334, memory: 35360, decode.loss_seg: 1.8553, decode.acc_seg: 36.8389, aux.loss_seg: 0.8613, aux.acc_seg: 31.1739, loss: 2.7166
2022-05-22 03:18:58,982 - mmseg - INFO - Iter [7650/320000]	lr: 5.857e-05, eta: 21:55:43, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7939, decode.acc_seg: 35.7929, aux.loss_seg: 0.8201, aux.acc_seg: 31.6425, loss: 2.6140
2022-05-22 03:19:11,476 - mmseg - INFO - Iter [7700/320000]	lr: 5.856e-05, eta: 21:55:25, time: 0.250, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9685, decode.acc_seg: 35.0026, aux.loss_seg: 0.9082, aux.acc_seg: 29.6573, loss: 2.8767
2022-05-22 03:19:23,788 - mmseg - INFO - Iter [7750/320000]	lr: 5.855e-05, eta: 21:54:59, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8158, decode.acc_seg: 36.7832, aux.loss_seg: 0.8588, aux.acc_seg: 30.0715, loss: 2.6746
2022-05-22 03:19:36,103 - mmseg - INFO - Iter [7800/320000]	lr: 5.854e-05, eta: 21:54:34, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8386, decode.acc_seg: 37.3121, aux.loss_seg: 0.8643, aux.acc_seg: 31.4421, loss: 2.7029
2022-05-22 03:19:48,470 - mmseg - INFO - Iter [7850/320000]	lr: 5.853e-05, eta: 21:54:11, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 1.9351, decode.acc_seg: 36.0757, aux.loss_seg: 0.8948, aux.acc_seg: 30.8657, loss: 2.8299
2022-05-22 03:20:00,734 - mmseg - INFO - Iter [7900/320000]	lr: 5.852e-05, eta: 21:53:43, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7664, decode.acc_seg: 36.8607, aux.loss_seg: 0.8344, aux.acc_seg: 31.1085, loss: 2.6008
2022-05-22 03:20:12,885 - mmseg - INFO - Iter [7950/320000]	lr: 5.851e-05, eta: 21:53:12, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7121, decode.acc_seg: 38.9964, aux.loss_seg: 0.8171, aux.acc_seg: 32.8865, loss: 2.5292
2022-05-22 03:20:25,346 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:20:25,346 - mmseg - INFO - Iter [8000/320000]	lr: 5.850e-05, eta: 21:52:53, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8198, decode.acc_seg: 37.0803, aux.loss_seg: 0.8393, aux.acc_seg: 32.0786, loss: 2.6591
2022-05-22 03:20:37,563 - mmseg - INFO - Iter [8050/320000]	lr: 5.849e-05, eta: 21:52:25, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.9127, decode.acc_seg: 35.8268, aux.loss_seg: 0.8801, aux.acc_seg: 30.7938, loss: 2.7927
2022-05-22 03:20:49,970 - mmseg - INFO - Iter [8100/320000]	lr: 5.848e-05, eta: 21:52:04, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7801, decode.acc_seg: 37.7833, aux.loss_seg: 0.8342, aux.acc_seg: 31.6036, loss: 2.6142
2022-05-22 03:21:02,515 - mmseg - INFO - Iter [8150/320000]	lr: 5.847e-05, eta: 21:51:48, time: 0.251, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7339, decode.acc_seg: 37.7660, aux.loss_seg: 0.8168, aux.acc_seg: 32.3149, loss: 2.5507
2022-05-22 03:21:14,791 - mmseg - INFO - Iter [8200/320000]	lr: 5.846e-05, eta: 21:51:23, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8473, decode.acc_seg: 36.4166, aux.loss_seg: 0.8618, aux.acc_seg: 31.5488, loss: 2.7090
2022-05-22 03:21:27,020 - mmseg - INFO - Iter [8250/320000]	lr: 5.845e-05, eta: 21:50:55, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7022, decode.acc_seg: 39.7062, aux.loss_seg: 0.8007, aux.acc_seg: 34.6360, loss: 2.5029
2022-05-22 03:21:39,442 - mmseg - INFO - Iter [8300/320000]	lr: 5.844e-05, eta: 21:50:35, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7801, decode.acc_seg: 35.5637, aux.loss_seg: 0.8301, aux.acc_seg: 30.2435, loss: 2.6102
2022-05-22 03:21:51,733 - mmseg - INFO - Iter [8350/320000]	lr: 5.843e-05, eta: 21:50:11, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7653, decode.acc_seg: 40.3613, aux.loss_seg: 0.8409, aux.acc_seg: 33.9226, loss: 2.6062
2022-05-22 03:22:04,244 - mmseg - INFO - Iter [8400/320000]	lr: 5.843e-05, eta: 21:49:54, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8022, decode.acc_seg: 37.9922, aux.loss_seg: 0.8503, aux.acc_seg: 32.5124, loss: 2.6525
2022-05-22 03:22:16,464 - mmseg - INFO - Iter [8450/320000]	lr: 5.842e-05, eta: 21:49:27, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7815, decode.acc_seg: 37.5042, aux.loss_seg: 0.8395, aux.acc_seg: 32.0472, loss: 2.6210
2022-05-22 03:22:28,752 - mmseg - INFO - Iter [8500/320000]	lr: 5.841e-05, eta: 21:49:03, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7042, decode.acc_seg: 39.7927, aux.loss_seg: 0.7975, aux.acc_seg: 34.3969, loss: 2.5017
2022-05-22 03:22:40,957 - mmseg - INFO - Iter [8550/320000]	lr: 5.840e-05, eta: 21:48:35, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8292, decode.acc_seg: 37.9947, aux.loss_seg: 0.8497, aux.acc_seg: 32.9199, loss: 2.6789
2022-05-22 03:22:53,259 - mmseg - INFO - Iter [8600/320000]	lr: 5.839e-05, eta: 21:48:12, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8245, decode.acc_seg: 37.0906, aux.loss_seg: 0.8497, aux.acc_seg: 31.6073, loss: 2.6742
2022-05-22 03:23:05,519 - mmseg - INFO - Iter [8650/320000]	lr: 5.838e-05, eta: 21:47:47, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7895, decode.acc_seg: 37.0585, aux.loss_seg: 0.8322, aux.acc_seg: 32.1047, loss: 2.6217
2022-05-22 03:23:17,789 - mmseg - INFO - Iter [8700/320000]	lr: 5.837e-05, eta: 21:47:22, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8138, decode.acc_seg: 35.5099, aux.loss_seg: 0.8418, aux.acc_seg: 30.2341, loss: 2.6556
2022-05-22 03:23:30,036 - mmseg - INFO - Iter [8750/320000]	lr: 5.836e-05, eta: 21:46:57, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7588, decode.acc_seg: 38.1580, aux.loss_seg: 0.8206, aux.acc_seg: 33.0238, loss: 2.5793
2022-05-22 03:23:42,308 - mmseg - INFO - Iter [8800/320000]	lr: 5.835e-05, eta: 21:46:33, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8048, decode.acc_seg: 37.3023, aux.loss_seg: 0.8253, aux.acc_seg: 33.1825, loss: 2.6301
2022-05-22 03:23:54,667 - mmseg - INFO - Iter [8850/320000]	lr: 5.834e-05, eta: 21:46:12, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8110, decode.acc_seg: 37.9784, aux.loss_seg: 0.8487, aux.acc_seg: 32.2255, loss: 2.6597
2022-05-22 03:24:07,228 - mmseg - INFO - Iter [8900/320000]	lr: 5.833e-05, eta: 21:45:59, time: 0.251, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7717, decode.acc_seg: 40.1953, aux.loss_seg: 0.8295, aux.acc_seg: 35.0192, loss: 2.6012
2022-05-22 03:24:19,661 - mmseg - INFO - Iter [8950/320000]	lr: 5.832e-05, eta: 21:45:40, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7209, decode.acc_seg: 36.4650, aux.loss_seg: 0.8142, aux.acc_seg: 31.0742, loss: 2.5351
2022-05-22 03:24:32,191 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:24:32,191 - mmseg - INFO - Iter [9000/320000]	lr: 5.831e-05, eta: 21:45:26, time: 0.251, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7831, decode.acc_seg: 36.4418, aux.loss_seg: 0.8266, aux.acc_seg: 31.7223, loss: 2.6097
2022-05-22 03:24:44,307 - mmseg - INFO - Iter [9050/320000]	lr: 5.830e-05, eta: 21:44:57, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7310, decode.acc_seg: 37.5158, aux.loss_seg: 0.8243, aux.acc_seg: 31.1285, loss: 2.5553
2022-05-22 03:24:56,534 - mmseg - INFO - Iter [9100/320000]	lr: 5.829e-05, eta: 21:44:32, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7079, decode.acc_seg: 37.6777, aux.loss_seg: 0.8280, aux.acc_seg: 30.7672, loss: 2.5359
2022-05-22 03:25:09,119 - mmseg - INFO - Iter [9150/320000]	lr: 5.828e-05, eta: 21:44:19, time: 0.252, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7550, decode.acc_seg: 37.5668, aux.loss_seg: 0.8168, aux.acc_seg: 32.7080, loss: 2.5718
2022-05-22 03:25:21,391 - mmseg - INFO - Iter [9200/320000]	lr: 5.828e-05, eta: 21:43:56, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6763, decode.acc_seg: 38.8034, aux.loss_seg: 0.7988, aux.acc_seg: 32.9004, loss: 2.4751
2022-05-22 03:25:33,749 - mmseg - INFO - Iter [9250/320000]	lr: 5.827e-05, eta: 21:43:35, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7801, decode.acc_seg: 37.4688, aux.loss_seg: 0.8327, aux.acc_seg: 32.5380, loss: 2.6129
2022-05-22 03:25:45,911 - mmseg - INFO - Iter [9300/320000]	lr: 5.826e-05, eta: 21:43:09, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6700, decode.acc_seg: 39.5835, aux.loss_seg: 0.7911, aux.acc_seg: 34.1995, loss: 2.4611
2022-05-22 03:25:58,106 - mmseg - INFO - Iter [9350/320000]	lr: 5.825e-05, eta: 21:42:43, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8020, decode.acc_seg: 38.1846, aux.loss_seg: 0.8426, aux.acc_seg: 33.0633, loss: 2.6445
2022-05-22 03:26:10,383 - mmseg - INFO - Iter [9400/320000]	lr: 5.824e-05, eta: 21:42:20, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7854, decode.acc_seg: 37.2531, aux.loss_seg: 0.8426, aux.acc_seg: 31.8167, loss: 2.6280
2022-05-22 03:26:22,498 - mmseg - INFO - Iter [9450/320000]	lr: 5.823e-05, eta: 21:41:53, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6550, decode.acc_seg: 38.3630, aux.loss_seg: 0.7840, aux.acc_seg: 33.0761, loss: 2.4390
2022-05-22 03:26:34,862 - mmseg - INFO - Iter [9500/320000]	lr: 5.822e-05, eta: 21:41:33, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7671, decode.acc_seg: 37.6285, aux.loss_seg: 0.8354, aux.acc_seg: 31.7965, loss: 2.6025
2022-05-22 03:26:47,129 - mmseg - INFO - Iter [9550/320000]	lr: 5.821e-05, eta: 21:41:10, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7268, decode.acc_seg: 38.7804, aux.loss_seg: 0.8105, aux.acc_seg: 33.4673, loss: 2.5373
2022-05-22 03:26:59,584 - mmseg - INFO - Iter [9600/320000]	lr: 5.820e-05, eta: 21:40:54, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8415, decode.acc_seg: 37.3966, aux.loss_seg: 0.8637, aux.acc_seg: 32.0078, loss: 2.7052
2022-05-22 03:27:11,849 - mmseg - INFO - Iter [9650/320000]	lr: 5.819e-05, eta: 21:40:32, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7240, decode.acc_seg: 38.8773, aux.loss_seg: 0.8160, aux.acc_seg: 33.2922, loss: 2.5400
2022-05-22 03:27:24,117 - mmseg - INFO - Iter [9700/320000]	lr: 5.818e-05, eta: 21:40:09, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7973, decode.acc_seg: 36.6686, aux.loss_seg: 0.8304, aux.acc_seg: 32.5658, loss: 2.6277
2022-05-22 03:27:36,344 - mmseg - INFO - Iter [9750/320000]	lr: 5.817e-05, eta: 21:39:46, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7304, decode.acc_seg: 37.3054, aux.loss_seg: 0.8101, aux.acc_seg: 31.4929, loss: 2.5404
2022-05-22 03:27:48,474 - mmseg - INFO - Iter [9800/320000]	lr: 5.816e-05, eta: 21:39:19, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7169, decode.acc_seg: 38.8016, aux.loss_seg: 0.8088, aux.acc_seg: 33.7907, loss: 2.5257
2022-05-22 03:28:00,803 - mmseg - INFO - Iter [9850/320000]	lr: 5.815e-05, eta: 21:38:59, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8310, decode.acc_seg: 37.4062, aux.loss_seg: 0.8494, aux.acc_seg: 32.2090, loss: 2.6804
2022-05-22 03:28:13,014 - mmseg - INFO - Iter [9900/320000]	lr: 5.814e-05, eta: 21:38:36, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8888, decode.acc_seg: 36.1451, aux.loss_seg: 0.8549, aux.acc_seg: 31.5417, loss: 2.7436
2022-05-22 03:28:25,353 - mmseg - INFO - Iter [9950/320000]	lr: 5.813e-05, eta: 21:38:16, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7154, decode.acc_seg: 39.2578, aux.loss_seg: 0.8220, aux.acc_seg: 32.7477, loss: 2.5374
2022-05-22 03:28:37,632 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:28:37,632 - mmseg - INFO - Iter [10000/320000]	lr: 5.813e-05, eta: 21:37:55, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7428, decode.acc_seg: 36.8956, aux.loss_seg: 0.8200, aux.acc_seg: 31.7582, loss: 2.5628
2022-05-22 03:28:49,736 - mmseg - INFO - Iter [10050/320000]	lr: 5.812e-05, eta: 21:37:28, time: 0.242, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8390, decode.acc_seg: 36.0023, aux.loss_seg: 0.8446, aux.acc_seg: 32.1318, loss: 2.6836
2022-05-22 03:29:02,250 - mmseg - INFO - Iter [10100/320000]	lr: 5.811e-05, eta: 21:37:14, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6236, decode.acc_seg: 39.8004, aux.loss_seg: 0.7647, aux.acc_seg: 35.7895, loss: 2.3883
2022-05-22 03:29:30,967 - mmseg - INFO - Iter [10150/320000]	lr: 5.810e-05, eta: 21:45:15, time: 0.574, data_time: 0.308, memory: 35360, decode.loss_seg: 1.7520, decode.acc_seg: 37.8551, aux.loss_seg: 0.8206, aux.acc_seg: 31.4284, loss: 2.5726
2022-05-22 03:29:43,601 - mmseg - INFO - Iter [10200/320000]	lr: 5.809e-05, eta: 21:45:02, time: 0.253, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7661, decode.acc_seg: 37.6360, aux.loss_seg: 0.8334, aux.acc_seg: 32.2405, loss: 2.5995
2022-05-22 03:29:56,020 - mmseg - INFO - Iter [10250/320000]	lr: 5.808e-05, eta: 21:44:43, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7036, decode.acc_seg: 37.9783, aux.loss_seg: 0.8021, aux.acc_seg: 33.1512, loss: 2.5057
2022-05-22 03:30:08,347 - mmseg - INFO - Iter [10300/320000]	lr: 5.807e-05, eta: 21:44:21, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7967, decode.acc_seg: 36.9603, aux.loss_seg: 0.8398, aux.acc_seg: 31.5799, loss: 2.6366
2022-05-22 03:30:20,790 - mmseg - INFO - Iter [10350/320000]	lr: 5.806e-05, eta: 21:44:02, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6982, decode.acc_seg: 40.0251, aux.loss_seg: 0.7975, aux.acc_seg: 34.4195, loss: 2.4957
2022-05-22 03:30:33,062 - mmseg - INFO - Iter [10400/320000]	lr: 5.805e-05, eta: 21:43:39, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7578, decode.acc_seg: 38.6414, aux.loss_seg: 0.8262, aux.acc_seg: 33.9902, loss: 2.5841
2022-05-22 03:30:45,500 - mmseg - INFO - Iter [10450/320000]	lr: 5.804e-05, eta: 21:43:21, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7555, decode.acc_seg: 39.0144, aux.loss_seg: 0.8339, aux.acc_seg: 32.3753, loss: 2.5894
2022-05-22 03:30:57,971 - mmseg - INFO - Iter [10500/320000]	lr: 5.803e-05, eta: 21:43:03, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7578, decode.acc_seg: 37.8519, aux.loss_seg: 0.8065, aux.acc_seg: 33.1830, loss: 2.5642
2022-05-22 03:31:10,287 - mmseg - INFO - Iter [10550/320000]	lr: 5.802e-05, eta: 21:42:41, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7329, decode.acc_seg: 37.8739, aux.loss_seg: 0.8120, aux.acc_seg: 32.8396, loss: 2.5450
2022-05-22 03:31:22,554 - mmseg - INFO - Iter [10600/320000]	lr: 5.801e-05, eta: 21:42:18, time: 0.245, data_time: 0.005, memory: 35360, decode.loss_seg: 1.7059, decode.acc_seg: 37.8974, aux.loss_seg: 0.7980, aux.acc_seg: 33.2887, loss: 2.5039
2022-05-22 03:31:34,808 - mmseg - INFO - Iter [10650/320000]	lr: 5.800e-05, eta: 21:41:55, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7817, decode.acc_seg: 38.1981, aux.loss_seg: 0.8400, aux.acc_seg: 32.5564, loss: 2.6217
2022-05-22 03:31:47,380 - mmseg - INFO - Iter [10700/320000]	lr: 5.799e-05, eta: 21:41:41, time: 0.251, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6968, decode.acc_seg: 37.8961, aux.loss_seg: 0.7989, aux.acc_seg: 32.7787, loss: 2.4957
2022-05-22 03:31:59,612 - mmseg - INFO - Iter [10750/320000]	lr: 5.798e-05, eta: 21:41:17, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7297, decode.acc_seg: 39.4124, aux.loss_seg: 0.8256, aux.acc_seg: 34.0694, loss: 2.5553
2022-05-22 03:32:12,341 - mmseg - INFO - Iter [10800/320000]	lr: 5.798e-05, eta: 21:41:07, time: 0.254, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6662, decode.acc_seg: 40.5108, aux.loss_seg: 0.7905, aux.acc_seg: 34.8844, loss: 2.4567
2022-05-22 03:32:24,752 - mmseg - INFO - Iter [10850/320000]	lr: 5.797e-05, eta: 21:40:48, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6452, decode.acc_seg: 40.6311, aux.loss_seg: 0.7790, aux.acc_seg: 34.5906, loss: 2.4242
2022-05-22 03:32:37,154 - mmseg - INFO - Iter [10900/320000]	lr: 5.796e-05, eta: 21:40:29, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7171, decode.acc_seg: 39.8551, aux.loss_seg: 0.8187, aux.acc_seg: 33.4269, loss: 2.5357
2022-05-22 03:32:49,505 - mmseg - INFO - Iter [10950/320000]	lr: 5.795e-05, eta: 21:40:09, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6815, decode.acc_seg: 38.5401, aux.loss_seg: 0.7961, aux.acc_seg: 33.2291, loss: 2.4776
2022-05-22 03:33:01,652 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:33:01,653 - mmseg - INFO - Iter [11000/320000]	lr: 5.794e-05, eta: 21:39:43, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 1.8006, decode.acc_seg: 38.7580, aux.loss_seg: 0.8558, aux.acc_seg: 33.1476, loss: 2.6563
2022-05-22 03:33:13,864 - mmseg - INFO - Iter [11050/320000]	lr: 5.793e-05, eta: 21:39:19, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7210, decode.acc_seg: 36.9797, aux.loss_seg: 0.8211, aux.acc_seg: 31.7728, loss: 2.5421
2022-05-22 03:33:26,193 - mmseg - INFO - Iter [11100/320000]	lr: 5.792e-05, eta: 21:38:58, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.8269, decode.acc_seg: 37.0550, aux.loss_seg: 0.8507, aux.acc_seg: 31.4396, loss: 2.6776
2022-05-22 03:33:38,643 - mmseg - INFO - Iter [11150/320000]	lr: 5.791e-05, eta: 21:38:41, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7553, decode.acc_seg: 38.0810, aux.loss_seg: 0.8298, aux.acc_seg: 31.9099, loss: 2.5852
2022-05-22 03:33:50,970 - mmseg - INFO - Iter [11200/320000]	lr: 5.790e-05, eta: 21:38:21, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7537, decode.acc_seg: 40.3199, aux.loss_seg: 0.8245, aux.acc_seg: 34.5047, loss: 2.5782
2022-05-22 03:34:03,214 - mmseg - INFO - Iter [11250/320000]	lr: 5.789e-05, eta: 21:37:58, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6263, decode.acc_seg: 39.4711, aux.loss_seg: 0.7870, aux.acc_seg: 33.2755, loss: 2.4133
2022-05-22 03:34:15,347 - mmseg - INFO - Iter [11300/320000]	lr: 5.788e-05, eta: 21:37:32, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6838, decode.acc_seg: 41.1991, aux.loss_seg: 0.7939, aux.acc_seg: 35.6857, loss: 2.4777
2022-05-22 03:34:27,559 - mmseg - INFO - Iter [11350/320000]	lr: 5.787e-05, eta: 21:37:09, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7091, decode.acc_seg: 39.0276, aux.loss_seg: 0.8091, aux.acc_seg: 33.4937, loss: 2.5182
2022-05-22 03:34:39,935 - mmseg - INFO - Iter [11400/320000]	lr: 5.786e-05, eta: 21:36:50, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6271, decode.acc_seg: 41.0668, aux.loss_seg: 0.7881, aux.acc_seg: 34.6746, loss: 2.4153
2022-05-22 03:34:52,154 - mmseg - INFO - Iter [11450/320000]	lr: 5.785e-05, eta: 21:36:27, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7557, decode.acc_seg: 36.9986, aux.loss_seg: 0.8278, aux.acc_seg: 31.4746, loss: 2.5836
2022-05-22 03:35:04,190 - mmseg - INFO - Iter [11500/320000]	lr: 5.784e-05, eta: 21:35:59, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7139, decode.acc_seg: 38.7050, aux.loss_seg: 0.8064, aux.acc_seg: 33.6740, loss: 2.5203
2022-05-22 03:35:16,540 - mmseg - INFO - Iter [11550/320000]	lr: 5.783e-05, eta: 21:35:39, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7986, decode.acc_seg: 38.2076, aux.loss_seg: 0.8323, aux.acc_seg: 33.8194, loss: 2.6309
2022-05-22 03:35:28,965 - mmseg - INFO - Iter [11600/320000]	lr: 5.783e-05, eta: 21:35:22, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6524, decode.acc_seg: 39.0498, aux.loss_seg: 0.7855, aux.acc_seg: 33.0095, loss: 2.4378
2022-05-22 03:35:41,375 - mmseg - INFO - Iter [11650/320000]	lr: 5.782e-05, eta: 21:35:05, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6997, decode.acc_seg: 40.6961, aux.loss_seg: 0.8159, aux.acc_seg: 34.4819, loss: 2.5157
2022-05-22 03:35:53,821 - mmseg - INFO - Iter [11700/320000]	lr: 5.781e-05, eta: 21:34:48, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7686, decode.acc_seg: 39.9275, aux.loss_seg: 0.8283, aux.acc_seg: 34.5926, loss: 2.5969
2022-05-22 03:36:06,232 - mmseg - INFO - Iter [11750/320000]	lr: 5.780e-05, eta: 21:34:30, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.5767, decode.acc_seg: 39.3975, aux.loss_seg: 0.7628, aux.acc_seg: 33.5037, loss: 2.3395
2022-05-22 03:36:18,267 - mmseg - INFO - Iter [11800/320000]	lr: 5.779e-05, eta: 21:34:03, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 1.5913, decode.acc_seg: 39.5884, aux.loss_seg: 0.7571, aux.acc_seg: 33.9626, loss: 2.3485
2022-05-22 03:36:30,778 - mmseg - INFO - Iter [11850/320000]	lr: 5.778e-05, eta: 21:33:48, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 1.5665, decode.acc_seg: 39.3870, aux.loss_seg: 0.7541, aux.acc_seg: 33.8959, loss: 2.3207
2022-05-22 03:36:43,299 - mmseg - INFO - Iter [11900/320000]	lr: 5.777e-05, eta: 21:33:34, time: 0.250, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7586, decode.acc_seg: 37.2435, aux.loss_seg: 0.8205, aux.acc_seg: 32.7053, loss: 2.5790
2022-05-22 03:36:55,588 - mmseg - INFO - Iter [11950/320000]	lr: 5.776e-05, eta: 21:33:13, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7806, decode.acc_seg: 37.1305, aux.loss_seg: 0.8282, aux.acc_seg: 31.5015, loss: 2.6089
2022-05-22 03:37:07,884 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:37:07,884 - mmseg - INFO - Iter [12000/320000]	lr: 5.775e-05, eta: 21:32:53, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6815, decode.acc_seg: 40.1686, aux.loss_seg: 0.8033, aux.acc_seg: 34.2571, loss: 2.4848
2022-05-22 03:37:20,234 - mmseg - INFO - Iter [12050/320000]	lr: 5.774e-05, eta: 21:32:34, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6197, decode.acc_seg: 40.5297, aux.loss_seg: 0.7751, aux.acc_seg: 34.7924, loss: 2.3949
2022-05-22 03:37:33,072 - mmseg - INFO - Iter [12100/320000]	lr: 5.773e-05, eta: 21:32:28, time: 0.257, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7444, decode.acc_seg: 39.4580, aux.loss_seg: 0.8226, aux.acc_seg: 33.0561, loss: 2.5670
2022-05-22 03:37:45,320 - mmseg - INFO - Iter [12150/320000]	lr: 5.772e-05, eta: 21:32:06, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6649, decode.acc_seg: 39.9490, aux.loss_seg: 0.7883, aux.acc_seg: 34.3133, loss: 2.4532
2022-05-22 03:37:57,623 - mmseg - INFO - Iter [12200/320000]	lr: 5.771e-05, eta: 21:31:47, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6960, decode.acc_seg: 38.5076, aux.loss_seg: 0.8006, aux.acc_seg: 32.7983, loss: 2.4966
2022-05-22 03:38:09,799 - mmseg - INFO - Iter [12250/320000]	lr: 5.770e-05, eta: 21:31:23, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7843, decode.acc_seg: 38.0081, aux.loss_seg: 0.8352, aux.acc_seg: 33.0753, loss: 2.6195
2022-05-22 03:38:22,111 - mmseg - INFO - Iter [12300/320000]	lr: 5.769e-05, eta: 21:31:04, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6588, decode.acc_seg: 39.3180, aux.loss_seg: 0.7910, aux.acc_seg: 32.8680, loss: 2.4498
2022-05-22 03:38:34,350 - mmseg - INFO - Iter [12350/320000]	lr: 5.768e-05, eta: 21:30:43, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6447, decode.acc_seg: 39.8994, aux.loss_seg: 0.7774, aux.acc_seg: 34.7338, loss: 2.4221
2022-05-22 03:38:46,528 - mmseg - INFO - Iter [12400/320000]	lr: 5.768e-05, eta: 21:30:20, time: 0.243, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6381, decode.acc_seg: 41.7317, aux.loss_seg: 0.7898, aux.acc_seg: 35.4519, loss: 2.4279
2022-05-22 03:38:58,910 - mmseg - INFO - Iter [12450/320000]	lr: 5.767e-05, eta: 21:30:02, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6573, decode.acc_seg: 39.2575, aux.loss_seg: 0.7781, aux.acc_seg: 33.7171, loss: 2.4355
2022-05-22 03:39:11,571 - mmseg - INFO - Iter [12500/320000]	lr: 5.766e-05, eta: 21:29:52, time: 0.253, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6989, decode.acc_seg: 40.1404, aux.loss_seg: 0.8016, aux.acc_seg: 34.7534, loss: 2.5006
2022-05-22 03:39:23,778 - mmseg - INFO - Iter [12550/320000]	lr: 5.765e-05, eta: 21:29:30, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6886, decode.acc_seg: 39.0132, aux.loss_seg: 0.8043, aux.acc_seg: 33.0888, loss: 2.4929
2022-05-22 03:39:36,149 - mmseg - INFO - Iter [12600/320000]	lr: 5.764e-05, eta: 21:29:12, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6565, decode.acc_seg: 40.8549, aux.loss_seg: 0.7899, aux.acc_seg: 35.4268, loss: 2.4464
2022-05-22 03:40:04,630 - mmseg - INFO - Iter [12650/320000]	lr: 5.763e-05, eta: 21:35:26, time: 0.570, data_time: 0.328, memory: 35360, decode.loss_seg: 1.7183, decode.acc_seg: 39.6554, aux.loss_seg: 0.8177, aux.acc_seg: 33.9690, loss: 2.5360
2022-05-22 03:40:16,914 - mmseg - INFO - Iter [12700/320000]	lr: 5.762e-05, eta: 21:35:04, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6708, decode.acc_seg: 38.8325, aux.loss_seg: 0.7941, aux.acc_seg: 33.3069, loss: 2.4649
2022-05-22 03:40:29,390 - mmseg - INFO - Iter [12750/320000]	lr: 5.761e-05, eta: 21:34:48, time: 0.250, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7103, decode.acc_seg: 39.3678, aux.loss_seg: 0.8144, aux.acc_seg: 33.2323, loss: 2.5247
2022-05-22 03:40:41,620 - mmseg - INFO - Iter [12800/320000]	lr: 5.760e-05, eta: 21:34:25, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6809, decode.acc_seg: 38.2068, aux.loss_seg: 0.7986, aux.acc_seg: 32.6351, loss: 2.4796
2022-05-22 03:40:53,913 - mmseg - INFO - Iter [12850/320000]	lr: 5.759e-05, eta: 21:34:04, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7455, decode.acc_seg: 38.5210, aux.loss_seg: 0.8201, aux.acc_seg: 32.9810, loss: 2.5656
2022-05-22 03:41:06,348 - mmseg - INFO - Iter [12900/320000]	lr: 5.758e-05, eta: 21:33:47, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6969, decode.acc_seg: 39.7145, aux.loss_seg: 0.8020, aux.acc_seg: 34.3727, loss: 2.4989
2022-05-22 03:41:18,540 - mmseg - INFO - Iter [12950/320000]	lr: 5.757e-05, eta: 21:33:23, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6906, decode.acc_seg: 38.8460, aux.loss_seg: 0.7849, aux.acc_seg: 32.9139, loss: 2.4755
2022-05-22 03:41:30,768 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:41:30,768 - mmseg - INFO - Iter [13000/320000]	lr: 5.756e-05, eta: 21:33:01, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7347, decode.acc_seg: 41.6703, aux.loss_seg: 0.8208, aux.acc_seg: 35.3002, loss: 2.5555
2022-05-22 03:41:43,159 - mmseg - INFO - Iter [13050/320000]	lr: 5.755e-05, eta: 21:32:43, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.5686, decode.acc_seg: 39.0025, aux.loss_seg: 0.7653, aux.acc_seg: 32.9184, loss: 2.3340
2022-05-22 03:41:55,629 - mmseg - INFO - Iter [13100/320000]	lr: 5.754e-05, eta: 21:32:26, time: 0.249, data_time: 0.005, memory: 35360, decode.loss_seg: 1.5668, decode.acc_seg: 41.1212, aux.loss_seg: 0.7458, aux.acc_seg: 35.7394, loss: 2.3126
2022-05-22 03:42:08,244 - mmseg - INFO - Iter [13150/320000]	lr: 5.753e-05, eta: 21:32:13, time: 0.252, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6368, decode.acc_seg: 40.0861, aux.loss_seg: 0.7782, aux.acc_seg: 34.3955, loss: 2.4150
2022-05-22 03:42:20,497 - mmseg - INFO - Iter [13200/320000]	lr: 5.753e-05, eta: 21:31:52, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6997, decode.acc_seg: 39.1642, aux.loss_seg: 0.7936, aux.acc_seg: 34.0691, loss: 2.4933
2022-05-22 03:42:32,829 - mmseg - INFO - Iter [13250/320000]	lr: 5.752e-05, eta: 21:31:32, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6409, decode.acc_seg: 40.4339, aux.loss_seg: 0.7771, aux.acc_seg: 34.9341, loss: 2.4180
2022-05-22 03:42:45,198 - mmseg - INFO - Iter [13300/320000]	lr: 5.751e-05, eta: 21:31:13, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6818, decode.acc_seg: 39.5370, aux.loss_seg: 0.8014, aux.acc_seg: 33.6626, loss: 2.4832
2022-05-22 03:42:57,482 - mmseg - INFO - Iter [13350/320000]	lr: 5.750e-05, eta: 21:30:53, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6447, decode.acc_seg: 39.9368, aux.loss_seg: 0.7838, aux.acc_seg: 34.0531, loss: 2.4285
2022-05-22 03:43:09,714 - mmseg - INFO - Iter [13400/320000]	lr: 5.749e-05, eta: 21:30:31, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6163, decode.acc_seg: 40.7824, aux.loss_seg: 0.7740, aux.acc_seg: 35.2460, loss: 2.3903
2022-05-22 03:43:21,745 - mmseg - INFO - Iter [13450/320000]	lr: 5.748e-05, eta: 21:30:05, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 1.5859, decode.acc_seg: 42.2082, aux.loss_seg: 0.7571, aux.acc_seg: 35.9734, loss: 2.3429
2022-05-22 03:43:34,024 - mmseg - INFO - Iter [13500/320000]	lr: 5.747e-05, eta: 21:29:44, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6973, decode.acc_seg: 40.2518, aux.loss_seg: 0.8110, aux.acc_seg: 34.1332, loss: 2.5083
2022-05-22 03:43:46,279 - mmseg - INFO - Iter [13550/320000]	lr: 5.746e-05, eta: 21:29:23, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6395, decode.acc_seg: 40.4674, aux.loss_seg: 0.7862, aux.acc_seg: 34.8886, loss: 2.4258
2022-05-22 03:43:58,791 - mmseg - INFO - Iter [13600/320000]	lr: 5.745e-05, eta: 21:29:08, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6103, decode.acc_seg: 40.4282, aux.loss_seg: 0.7686, aux.acc_seg: 34.9005, loss: 2.3789
2022-05-22 03:44:11,106 - mmseg - INFO - Iter [13650/320000]	lr: 5.744e-05, eta: 21:28:49, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6923, decode.acc_seg: 40.0482, aux.loss_seg: 0.8051, aux.acc_seg: 33.7367, loss: 2.4975
2022-05-22 03:44:23,272 - mmseg - INFO - Iter [13700/320000]	lr: 5.743e-05, eta: 21:28:26, time: 0.243, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6740, decode.acc_seg: 40.3947, aux.loss_seg: 0.7965, aux.acc_seg: 34.1487, loss: 2.4705
2022-05-22 03:44:35,707 - mmseg - INFO - Iter [13750/320000]	lr: 5.742e-05, eta: 21:28:09, time: 0.249, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6491, decode.acc_seg: 40.8862, aux.loss_seg: 0.7692, aux.acc_seg: 35.1708, loss: 2.4183
2022-05-22 03:44:47,935 - mmseg - INFO - Iter [13800/320000]	lr: 5.741e-05, eta: 21:27:48, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6449, decode.acc_seg: 40.4545, aux.loss_seg: 0.7862, aux.acc_seg: 34.5604, loss: 2.4311
2022-05-22 03:45:00,416 - mmseg - INFO - Iter [13850/320000]	lr: 5.740e-05, eta: 21:27:32, time: 0.250, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6320, decode.acc_seg: 39.1412, aux.loss_seg: 0.7722, aux.acc_seg: 34.2645, loss: 2.4043
2022-05-22 03:45:12,615 - mmseg - INFO - Iter [13900/320000]	lr: 5.739e-05, eta: 21:27:10, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6662, decode.acc_seg: 39.5260, aux.loss_seg: 0.7866, aux.acc_seg: 34.6350, loss: 2.4528
2022-05-22 03:45:25,015 - mmseg - INFO - Iter [13950/320000]	lr: 5.738e-05, eta: 21:26:53, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6484, decode.acc_seg: 39.0290, aux.loss_seg: 0.7834, aux.acc_seg: 33.6323, loss: 2.4318
2022-05-22 03:45:37,419 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:45:37,419 - mmseg - INFO - Iter [14000/320000]	lr: 5.738e-05, eta: 21:26:36, time: 0.248, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6818, decode.acc_seg: 39.4348, aux.loss_seg: 0.8029, aux.acc_seg: 33.1259, loss: 2.4847
2022-05-22 03:45:49,767 - mmseg - INFO - Iter [14050/320000]	lr: 5.737e-05, eta: 21:26:17, time: 0.247, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6214, decode.acc_seg: 41.3896, aux.loss_seg: 0.7799, aux.acc_seg: 34.8722, loss: 2.4014
2022-05-22 03:46:01,953 - mmseg - INFO - Iter [14100/320000]	lr: 5.736e-05, eta: 21:25:55, time: 0.244, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6749, decode.acc_seg: 40.9029, aux.loss_seg: 0.7955, aux.acc_seg: 35.5482, loss: 2.4704
2022-05-22 03:46:14,261 - mmseg - INFO - Iter [14150/320000]	lr: 5.735e-05, eta: 21:25:36, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6996, decode.acc_seg: 39.0199, aux.loss_seg: 0.7949, aux.acc_seg: 34.4380, loss: 2.4945
2022-05-22 03:46:26,509 - mmseg - INFO - Iter [14200/320000]	lr: 5.734e-05, eta: 21:25:16, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6832, decode.acc_seg: 39.7391, aux.loss_seg: 0.8035, aux.acc_seg: 33.8465, loss: 2.4867
2022-05-22 03:46:38,802 - mmseg - INFO - Iter [14250/320000]	lr: 5.733e-05, eta: 21:24:57, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7026, decode.acc_seg: 39.6178, aux.loss_seg: 0.8026, aux.acc_seg: 33.5558, loss: 2.5052
2022-05-22 03:46:51,139 - mmseg - INFO - Iter [14300/320000]	lr: 5.732e-05, eta: 21:24:38, time: 0.247, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6174, decode.acc_seg: 40.5475, aux.loss_seg: 0.7776, aux.acc_seg: 34.7251, loss: 2.3950
2022-05-22 03:47:03,438 - mmseg - INFO - Iter [14350/320000]	lr: 5.731e-05, eta: 21:24:19, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6447, decode.acc_seg: 40.3712, aux.loss_seg: 0.7696, aux.acc_seg: 35.7073, loss: 2.4143
2022-05-22 03:47:15,609 - mmseg - INFO - Iter [14400/320000]	lr: 5.730e-05, eta: 21:23:57, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.7045, decode.acc_seg: 40.1048, aux.loss_seg: 0.8127, aux.acc_seg: 33.7703, loss: 2.5171
2022-05-22 03:47:27,861 - mmseg - INFO - Iter [14450/320000]	lr: 5.729e-05, eta: 21:23:37, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6162, decode.acc_seg: 40.7400, aux.loss_seg: 0.7682, aux.acc_seg: 35.3401, loss: 2.3844
2022-05-22 03:47:40,247 - mmseg - INFO - Iter [14500/320000]	lr: 5.728e-05, eta: 21:23:20, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6324, decode.acc_seg: 42.4834, aux.loss_seg: 0.7841, aux.acc_seg: 36.7948, loss: 2.4165
2022-05-22 03:47:52,558 - mmseg - INFO - Iter [14550/320000]	lr: 5.727e-05, eta: 21:23:01, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6197, decode.acc_seg: 39.0657, aux.loss_seg: 0.7705, aux.acc_seg: 33.5295, loss: 2.3901
2022-05-22 03:48:04,817 - mmseg - INFO - Iter [14600/320000]	lr: 5.726e-05, eta: 21:22:41, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6536, decode.acc_seg: 40.5418, aux.loss_seg: 0.7908, aux.acc_seg: 34.1621, loss: 2.4444
2022-05-22 03:48:16,928 - mmseg - INFO - Iter [14650/320000]	lr: 5.725e-05, eta: 21:22:19, time: 0.242, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6840, decode.acc_seg: 39.5536, aux.loss_seg: 0.7974, aux.acc_seg: 34.0439, loss: 2.4814
2022-05-22 03:48:29,132 - mmseg - INFO - Iter [14700/320000]	lr: 5.724e-05, eta: 21:21:58, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 1.5778, decode.acc_seg: 43.7275, aux.loss_seg: 0.7601, aux.acc_seg: 38.1721, loss: 2.3379
2022-05-22 03:48:41,379 - mmseg - INFO - Iter [14750/320000]	lr: 5.723e-05, eta: 21:21:38, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6119, decode.acc_seg: 39.0436, aux.loss_seg: 0.7754, aux.acc_seg: 33.5004, loss: 2.3873
2022-05-22 03:48:53,612 - mmseg - INFO - Iter [14800/320000]	lr: 5.723e-05, eta: 21:21:18, time: 0.245, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6519, decode.acc_seg: 41.0774, aux.loss_seg: 0.7921, aux.acc_seg: 35.0442, loss: 2.4440
2022-05-22 03:49:05,908 - mmseg - INFO - Iter [14850/320000]	lr: 5.722e-05, eta: 21:20:59, time: 0.246, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6629, decode.acc_seg: 37.9908, aux.loss_seg: 0.7738, aux.acc_seg: 33.2167, loss: 2.4368
2022-05-22 03:49:18,184 - mmseg - INFO - Iter [14900/320000]	lr: 5.721e-05, eta: 21:20:40, time: 0.245, data_time: 0.004, memory: 35360, decode.loss_seg: 1.7353, decode.acc_seg: 39.0697, aux.loss_seg: 0.8116, aux.acc_seg: 34.2972, loss: 2.5469
2022-05-22 03:49:30,587 - mmseg - INFO - Iter [14950/320000]	lr: 5.720e-05, eta: 21:20:24, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6406, decode.acc_seg: 39.6848, aux.loss_seg: 0.7767, aux.acc_seg: 32.7403, loss: 2.4173
2022-05-22 03:49:42,978 - mmseg - INFO - Exp name: bisa_baseline.py
2022-05-22 03:49:42,978 - mmseg - INFO - Iter [15000/320000]	lr: 5.719e-05, eta: 21:20:07, time: 0.248, data_time: 0.005, memory: 35360, decode.loss_seg: 1.5402, decode.acc_seg: 40.3050, aux.loss_seg: 0.7328, aux.acc_seg: 35.2329, loss: 2.2730
2022-05-22 03:49:55,410 - mmseg - INFO - Iter [15050/320000]	lr: 5.718e-05, eta: 21:19:51, time: 0.249, data_time: 0.004, memory: 35360, decode.loss_seg: 1.6033, decode.acc_seg: 40.6881, aux.loss_seg: 0.7635, aux.acc_seg: 35.5968, loss: 2.3668
2022-05-22 03:50:07,791 - mmseg - INFO - Iter [15100/320000]	lr: 5.717e-05, eta: 21:19:34, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 1.6107, decode.acc_seg: 42.6788, aux.loss_seg: 0.7750, aux.acc_seg: 36.3647, loss: 2.3857
2022-05-22 03:50:20,093 - mmseg - INFO - Iter [15150/320000]	lr: 5.716e-05, eta: 21:19:16, time: 0.246, data_time: 0.003, memory: 35360, decode.loss_seg: 1.5627, decode.acc_seg: 42.0169, aux.loss_seg: 0.7415, aux.acc_seg: 36.6205, loss: 2.3042
Traceback (most recent call last):
  File "<string>", line 1, in <module>
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 116, in spawn_main
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 125, in _main
    exitcode = _main(fd, parent_sentinel)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 125, in _main
    prepare(preparation_data)
    prepare(preparation_data)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 236, in prepare
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
    _fixup_main_from_path(data['init_main_from_path'])
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 287, in _fixup_main_from_path
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 268, in run_path
    main_content = runpy.run_path(main_path,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 268, in run_path
    return _run_module_code(code, init_globals, run_name,
    return _run_module_code(code, init_globals, run_name,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 97, in _run_module_code
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 97, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 87, in _run_code
    _run_code(code, mod_globals, init_globals,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
    exec(code, run_globals)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 13, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 13, in <module>
    from .bisa_transformer import BisaSwinTransformer
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 17, in <module>
    from .bisa_transformer import BisaSwinTransformer
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 17, in <module>
    from .bisa import BiDirectionalWindowAttention
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py", line 65
    lambda_req_grad=True
    ^
SyntaxError: invalid syntax
    from .bisa import BiDirectionalWindowAttention
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py", line 65
    lambda_req_grad=True
    ^
SyntaxError: invalid syntax
Traceback (most recent call last):
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 116, in spawn_main
  File "<string>", line 1, in <module>
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 125, in _main
    exitcode = _main(fd, parent_sentinel)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 125, in _main
    prepare(preparation_data)
    prepare(preparation_data)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 236, in prepare
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 287, in _fixup_main_from_path
    _fixup_main_from_path(data['init_main_from_path'])
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/multiprocessing/spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 268, in run_path
    main_content = runpy.run_path(main_path,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 268, in run_path
    return _run_module_code(code, init_globals, run_name,
    return _run_module_code(code, init_globals, run_name,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 97, in _run_module_code
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 97, in _run_module_code
    _run_code(code, mod_globals, init_globals,
    _run_code(code, mod_globals, init_globals,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 87, in _run_code
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
    exec(code, run_globals)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 13, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from .bisa_transformer import BisaSwinTransformer
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 17, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 13, in <module>
    from .bisa_transformer import BisaSwinTransformer
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 17, in <module>
    from .bisa import BiDirectionalWindowAttention
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py", line 65
    lambda_req_grad=True
    ^
    from .bisa import BiDirectionalWindowAttention
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py", line 65
    lambda_req_grad=True
    ^
SyntaxError: invalid syntax
SyntaxError: invalid syntax
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
    from .bisa_transformer_ng import BisaSwinTransformerNg
ModuleNotFoundError: No module named 'mmseg.models.backbones.bisa_transformer_ng'
    from .bisa_transformer_ng import BisaSwinTransformerNg
ModuleNotFoundError: No module named 'mmseg.models.backbones.bisa_transformer_ng'
Traceback (most recent call last):
Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from .backbones import *  # noqa: F401,F403
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
        from .bisa_transformer_ng import BisaSwinTransformerNg
ModuleNotFoundError: No module named 'mmseg.models.backbones.bisa_transformer_ng'
from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
    from .bisa_transformer_ng import BisaSwinTransformerNg
ModuleNotFoundError: No module named 'mmseg.models.backbones.bisa_transformer_ng'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 27444) of binary: /nethome/bdevnani3/flash1/envs/mmlab/bin/python
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py", line 193, in <module>
    main()
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py", line 189, in main
    launch(args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py", line 174, in launch
    run(args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/run.py", line 715, in run
    elastic_launch(
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 131, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 245, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
tools/train.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2022-05-22_03:56:30
  host      : sonny.cc.gatech.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 27445)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2022-05-22_03:56:30
  host      : sonny.cc.gatech.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 27446)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2022-05-22_03:56:30
  host      : sonny.cc.gatech.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 27447)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2022-05-22_03:56:30
  host      : sonny.cc.gatech.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 27444)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:56:48,299 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: A40
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

2022-05-22 03:56:48,300 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:56:48,840 - mmseg - INFO - Config:
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
        reverse_attention_locations=[],
        apply_bidirectional_layer_norms=False,
        bidirectional_lambda_value=-100.0,
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
work_dir = './work_dirs/bisa_baseline'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:56:51,613 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 03:56:52,019 - mmseg - INFO - Loaded 20210 images
2022-05-22 03:56:55,431 - mmseg - INFO - Loaded 2000 images
2022-05-22 03:56:55,431 - mmseg - INFO - Start running, host: bdevnani3@sonny, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_baseline
2022-05-22 03:56:55,432 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 03:56:55,432 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 03:56:55,432 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_baseline by HardDiskBackend.
2022-05-22 03:57:12,032 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 03:57:15,946 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
2022-05-22 03:57:28,068 - mmseg - INFO - Iter [50/320000]	lr: 9.799e-07, eta: 1 day, 11:30:00, time: 0.399, data_time: 0.008, memory: 35360, decode.loss_seg: 4.1348, decode.acc_seg: 0.5655, aux.loss_seg: 1.6581, aux.acc_seg: 0.4805, loss: 5.7930
2022-05-22 03:57:40,448 - mmseg - INFO - Iter [100/320000]	lr: 1.979e-06, eta: 1 day, 4:44:47, time: 0.248, data_time: 0.003, memory: 35360, decode.loss_seg: 4.0628, decode.acc_seg: 1.6593, aux.loss_seg: 1.6346, aux.acc_seg: 0.6875, loss: 5.6973
2022-05-22 03:57:52,662 - mmseg - INFO - Iter [150/320000]	lr: 2.979e-06, eta: 1 day, 2:23:46, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 4.0138, decode.acc_seg: 6.4086, aux.loss_seg: 1.6284, aux.acc_seg: 0.9384, loss: 5.6422
2022-05-22 03:58:04,844 - mmseg - INFO - Iter [200/320000]	lr: 3.978e-06, eta: 1 day, 1:12:08, time: 0.244, data_time: 0.003, memory: 35360, decode.loss_seg: 3.9215, decode.acc_seg: 10.6301, aux.loss_seg: 1.6092, aux.acc_seg: 1.1228, loss: 5.5308
2022-05-22 03:58:16,905 - mmseg - INFO - Iter [250/320000]	lr: 4.976e-06, eta: 1 day, 0:26:42, time: 0.241, data_time: 0.003, memory: 35360, decode.loss_seg: 3.8224, decode.acc_seg: 13.5078, aux.loss_seg: 1.5889, aux.acc_seg: 1.7668, loss: 5.4114
