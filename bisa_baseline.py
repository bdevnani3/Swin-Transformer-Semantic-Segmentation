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
