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
