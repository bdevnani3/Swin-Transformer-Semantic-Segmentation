/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
[W socket.cpp:401] [c10d] The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use).
[W socket.cpp:401] [c10d] The server socket has failed to bind to 0.0.0.0:29500 (errno: 98 - Address already in use).
[E socket.cpp:435] [c10d] The server socket has failed to listen on any local network address.
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
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 236, in launch_agent
    result = agent.run()
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/metrics/api.py", line 125, in wrapper
    result = f(*args, **kwargs)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/agent/server/api.py", line 709, in run
    result = self._invoke_run(role)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/agent/server/api.py", line 844, in _invoke_run
    self._initialize_workers(self._worker_group)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/metrics/api.py", line 125, in wrapper
    result = f(*args, **kwargs)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/agent/server/api.py", line 678, in _initialize_workers
    self._rendezvous(worker_group)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/metrics/api.py", line 125, in wrapper
    result = f(*args, **kwargs)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/agent/server/api.py", line 538, in _rendezvous
    store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py", line 55, in next_rendezvous
    self._store = TCPStore(  # type: ignore[call-arg]
RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to 0.0.0.0:29500 (errno: 98 - Address already in use).
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
2022-05-22 03:38:28,676 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: GeForce RTX 2080 Ti
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

2022-05-22 03:38:28,676 - mmseg - INFO - Distributed training: True
2022-05-22 03:38:28,961 - mmseg - INFO - Config:
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
        apply_bidirectional_layer_norms=True,
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
work_dir = './work_dirs/bisa_norm_learned'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:38:30,537 - mmseg - INFO - EncoderDecoder(
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
              (msa_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=False)
              (isa_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=False)
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
              (msa_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=False)
              (isa_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=False)
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
2022-05-22 03:38:32,428 - mmseg - INFO - Loaded 20210 images
2022-05-22 03:38:37,494 - mmseg - INFO - Loaded 2000 images
2022-05-22 03:38:37,494 - mmseg - INFO - Start running, host: bdevnani3@chomps, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_learned
2022-05-22 03:38:37,495 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 03:38:37,495 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 03:38:37,495 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_learned by HardDiskBackend.
2022-05-22 03:38:52,241 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 03:38:53,957 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
2022-05-22 03:39:14,809 - mmseg - INFO - Iter [50/320000]	lr: 9.799e-07, eta: 1 day, 21:58:25, time: 0.517, data_time: 0.013, memory: 8044, decode.loss_seg: 4.0367, decode.acc_seg: 0.3043, aux.loss_seg: 1.6125, aux.acc_seg: 0.4032, loss: 5.6491
2022-05-22 03:39:36,127 - mmseg - INFO - Iter [100/320000]	lr: 1.979e-06, eta: 1 day, 17:55:33, time: 0.426, data_time: 0.004, memory: 8044, decode.loss_seg: 4.0538, decode.acc_seg: 0.5957, aux.loss_seg: 1.6265, aux.acc_seg: 0.4166, loss: 5.6802
2022-05-22 03:39:57,444 - mmseg - INFO - Iter [150/320000]	lr: 2.979e-06, eta: 1 day, 16:34:01, time: 0.426, data_time: 0.004, memory: 8044, decode.loss_seg: 4.0605, decode.acc_seg: 3.5396, aux.loss_seg: 1.6427, aux.acc_seg: 1.2602, loss: 5.7033
2022-05-22 03:40:18,741 - mmseg - INFO - Iter [200/320000]	lr: 3.978e-06, eta: 1 day, 15:53:02, time: 0.426, data_time: 0.004, memory: 8044, decode.loss_seg: 3.8752, decode.acc_seg: 9.1793, aux.loss_seg: 1.5889, aux.acc_seg: 3.9798, loss: 5.4641
2022-05-22 03:40:39,871 - mmseg - INFO - Iter [250/320000]	lr: 4.976e-06, eta: 1 day, 15:24:32, time: 0.423, data_time: 0.004, memory: 8044, decode.loss_seg: 3.8217, decode.acc_seg: 12.8345, aux.loss_seg: 1.5847, aux.acc_seg: 5.4715, loss: 5.4064
2022-05-22 03:41:01,068 - mmseg - INFO - Iter [300/320000]	lr: 5.974e-06, eta: 1 day, 15:06:28, time: 0.424, data_time: 0.005, memory: 8044, decode.loss_seg: 3.8415, decode.acc_seg: 14.4529, aux.loss_seg: 1.6052, aux.acc_seg: 6.0760, loss: 5.4468
2022-05-22 03:41:22,307 - mmseg - INFO - Iter [350/320000]	lr: 6.972e-06, eta: 1 day, 14:54:16, time: 0.425, data_time: 0.004, memory: 8044, decode.loss_seg: 3.8629, decode.acc_seg: 17.0615, aux.loss_seg: 1.6390, aux.acc_seg: 8.4173, loss: 5.5018
2022-05-22 03:41:43,622 - mmseg - INFO - Iter [400/320000]	lr: 7.970e-06, eta: 1 day, 14:46:04, time: 0.426, data_time: 0.004, memory: 8044, decode.loss_seg: 3.6404, decode.acc_seg: 18.2555, aux.loss_seg: 1.5609, aux.acc_seg: 11.3246, loss: 5.2014
2022-05-22 03:42:04,742 - mmseg - INFO - Iter [450/320000]	lr: 8.967e-06, eta: 1 day, 14:37:10, time: 0.422, data_time: 0.004, memory: 8044, decode.loss_seg: 3.5310, decode.acc_seg: 17.6485, aux.loss_seg: 1.5308, aux.acc_seg: 11.9943, loss: 5.0619
2022-05-22 03:42:25,847 - mmseg - INFO - Iter [500/320000]	lr: 9.964e-06, eta: 1 day, 14:29:53, time: 0.422, data_time: 0.004, memory: 8044, decode.loss_seg: 3.5758, decode.acc_seg: 18.3443, aux.loss_seg: 1.5540, aux.acc_seg: 13.9635, loss: 5.1299
2022-05-22 03:42:47,005 - mmseg - INFO - Iter [550/320000]	lr: 1.096e-05, eta: 1 day, 14:24:27, time: 0.423, data_time: 0.005, memory: 8044, decode.loss_seg: 3.5863, decode.acc_seg: 19.4063, aux.loss_seg: 1.5708, aux.acc_seg: 13.8106, loss: 5.1571
2022-05-22 03:43:07,880 - mmseg - INFO - Iter [600/320000]	lr: 1.196e-05, eta: 1 day, 14:17:13, time: 0.417, data_time: 0.004, memory: 8044, decode.loss_seg: 3.4347, decode.acc_seg: 19.9312, aux.loss_seg: 1.5124, aux.acc_seg: 15.1058, loss: 4.9471
2022-05-22 03:43:28,945 - mmseg - INFO - Iter [650/320000]	lr: 1.295e-05, eta: 1 day, 14:12:41, time: 0.421, data_time: 0.004, memory: 8044, decode.loss_seg: 3.5138, decode.acc_seg: 22.5175, aux.loss_seg: 1.5599, aux.acc_seg: 17.5639, loss: 5.0737
2022-05-22 03:43:49,910 - mmseg - INFO - Iter [700/320000]	lr: 1.395e-05, eta: 1 day, 14:07:57, time: 0.419, data_time: 0.004, memory: 8044, decode.loss_seg: 3.2478, decode.acc_seg: 22.3477, aux.loss_seg: 1.4652, aux.acc_seg: 17.6828, loss: 4.7129
2022-05-22 03:44:10,848 - mmseg - INFO - Iter [750/320000]	lr: 1.494e-05, eta: 1 day, 14:03:37, time: 0.419, data_time: 0.004, memory: 8044, decode.loss_seg: 3.3343, decode.acc_seg: 21.3721, aux.loss_seg: 1.4938, aux.acc_seg: 17.5595, loss: 4.8281
2022-05-22 03:44:31,695 - mmseg - INFO - Iter [800/320000]	lr: 1.594e-05, eta: 1 day, 13:59:11, time: 0.417, data_time: 0.004, memory: 8044, decode.loss_seg: 3.2644, decode.acc_seg: 23.5867, aux.loss_seg: 1.4836, aux.acc_seg: 18.9932, loss: 4.7480
2022-05-22 03:44:52,465 - mmseg - INFO - Iter [850/320000]	lr: 1.693e-05, eta: 1 day, 13:54:46, time: 0.415, data_time: 0.004, memory: 8044, decode.loss_seg: 3.1803, decode.acc_seg: 21.9270, aux.loss_seg: 1.4518, aux.acc_seg: 18.5569, loss: 4.6321
2022-05-22 03:45:13,531 - mmseg - INFO - Iter [900/320000]	lr: 1.793e-05, eta: 1 day, 13:52:34, time: 0.421, data_time: 0.004, memory: 8044, decode.loss_seg: 3.0864, decode.acc_seg: 22.4810, aux.loss_seg: 1.4144, aux.acc_seg: 18.5816, loss: 4.5008
2022-05-22 03:45:34,418 - mmseg - INFO - Iter [950/320000]	lr: 1.892e-05, eta: 1 day, 13:49:30, time: 0.418, data_time: 0.004, memory: 8044, decode.loss_seg: 3.0393, decode.acc_seg: 22.2127, aux.loss_seg: 1.4062, aux.acc_seg: 18.7347, loss: 4.4455
2022-05-22 03:45:55,408 - mmseg - INFO - Exp name: bisa_norm_learned.py
2022-05-22 03:45:55,408 - mmseg - INFO - Iter [1000/320000]	lr: 1.992e-05, eta: 1 day, 13:47:17, time: 0.420, data_time: 0.004, memory: 8044, decode.loss_seg: 3.0426, decode.acc_seg: 23.0555, aux.loss_seg: 1.4167, aux.acc_seg: 18.4726, loss: 4.4593
2022-05-22 03:46:16,379 - mmseg - INFO - Iter [1050/320000]	lr: 2.091e-05, eta: 1 day, 13:45:10, time: 0.420, data_time: 0.004, memory: 8044, decode.loss_seg: 2.9569, decode.acc_seg: 24.8793, aux.loss_seg: 1.3909, aux.acc_seg: 19.7797, loss: 4.3478
2022-05-22 03:46:37,422 - mmseg - INFO - Iter [1100/320000]	lr: 2.190e-05, eta: 1 day, 13:43:32, time: 0.421, data_time: 0.004, memory: 8044, decode.loss_seg: 3.0064, decode.acc_seg: 24.0354, aux.loss_seg: 1.4021, aux.acc_seg: 20.4267, loss: 4.4085
2022-05-22 03:46:58,316 - mmseg - INFO - Iter [1150/320000]	lr: 2.290e-05, eta: 1 day, 13:41:20, time: 0.418, data_time: 0.004, memory: 8044, decode.loss_seg: 2.9131, decode.acc_seg: 23.7524, aux.loss_seg: 1.3665, aux.acc_seg: 20.5272, loss: 4.2796
2022-05-22 03:47:19,096 - mmseg - INFO - Iter [1200/320000]	lr: 2.389e-05, eta: 1 day, 13:38:47, time: 0.416, data_time: 0.004, memory: 8044, decode.loss_seg: 2.7535, decode.acc_seg: 24.4014, aux.loss_seg: 1.3147, aux.acc_seg: 19.7105, loss: 4.0682
2022-05-22 03:47:40,072 - mmseg - INFO - Iter [1250/320000]	lr: 2.488e-05, eta: 1 day, 13:37:13, time: 0.419, data_time: 0.004, memory: 8044, decode.loss_seg: 2.7864, decode.acc_seg: 25.9332, aux.loss_seg: 1.3308, aux.acc_seg: 21.2302, loss: 4.1172
2022-05-22 03:48:00,877 - mmseg - INFO - Iter [1300/320000]	lr: 2.587e-05, eta: 1 day, 13:35:04, time: 0.416, data_time: 0.004, memory: 8044, decode.loss_seg: 2.7485, decode.acc_seg: 24.0709, aux.loss_seg: 1.3042, aux.acc_seg: 20.1582, loss: 4.0527
2022-05-22 03:48:21,670 - mmseg - INFO - Iter [1350/320000]	lr: 2.687e-05, eta: 1 day, 13:33:01, time: 0.416, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6997, decode.acc_seg: 23.8108, aux.loss_seg: 1.2760, aux.acc_seg: 19.5490, loss: 3.9757
2022-05-22 03:48:42,399 - mmseg - INFO - Iter [1400/320000]	lr: 2.786e-05, eta: 1 day, 13:30:50, time: 0.415, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6780, decode.acc_seg: 26.4955, aux.loss_seg: 1.2858, aux.acc_seg: 21.8142, loss: 3.9638
2022-05-22 03:49:03,182 - mmseg - INFO - Iter [1450/320000]	lr: 2.885e-05, eta: 1 day, 13:28:57, time: 0.416, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6778, decode.acc_seg: 25.1698, aux.loss_seg: 1.2733, aux.acc_seg: 21.3890, loss: 3.9511
2022-05-22 03:49:23,981 - mmseg - INFO - Iter [1500/320000]	lr: 2.984e-05, eta: 1 day, 13:27:15, time: 0.416, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6234, decode.acc_seg: 23.9200, aux.loss_seg: 1.2294, aux.acc_seg: 20.4366, loss: 3.8528
2022-05-22 03:49:44,954 - mmseg - INFO - Iter [1550/320000]	lr: 3.083e-05, eta: 1 day, 13:26:15, time: 0.419, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6110, decode.acc_seg: 27.5952, aux.loss_seg: 1.2476, aux.acc_seg: 22.7637, loss: 3.8586
2022-05-22 03:50:05,966 - mmseg - INFO - Iter [1600/320000]	lr: 3.182e-05, eta: 1 day, 13:25:24, time: 0.420, data_time: 0.004, memory: 8044, decode.loss_seg: 2.4821, decode.acc_seg: 27.1221, aux.loss_seg: 1.1776, aux.acc_seg: 22.5168, loss: 3.6597
2022-05-22 03:50:26,851 - mmseg - INFO - Iter [1650/320000]	lr: 3.281e-05, eta: 1 day, 13:24:09, time: 0.418, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6100, decode.acc_seg: 23.9157, aux.loss_seg: 1.2164, aux.acc_seg: 19.8261, loss: 3.8264
2022-05-22 03:50:47,668 - mmseg - INFO - Iter [1700/320000]	lr: 3.380e-05, eta: 1 day, 13:22:47, time: 0.416, data_time: 0.005, memory: 8044, decode.loss_seg: 2.5491, decode.acc_seg: 27.6085, aux.loss_seg: 1.2053, aux.acc_seg: 22.7555, loss: 3.7544
2022-05-22 03:51:08,563 - mmseg - INFO - Iter [1750/320000]	lr: 3.479e-05, eta: 1 day, 13:21:41, time: 0.418, data_time: 0.005, memory: 8044, decode.loss_seg: 2.4393, decode.acc_seg: 28.5146, aux.loss_seg: 1.1596, aux.acc_seg: 23.4946, loss: 3.5988
2022-05-22 03:51:29,432 - mmseg - INFO - Iter [1800/320000]	lr: 3.578e-05, eta: 1 day, 13:20:33, time: 0.417, data_time: 0.004, memory: 8044, decode.loss_seg: 2.5055, decode.acc_seg: 26.3390, aux.loss_seg: 1.1741, aux.acc_seg: 21.9104, loss: 3.6796
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
Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
        from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from mmseg.models import build_segmentor
    from mmseg.models import build_segmentor  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>

      File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
        from .backbones import *  # noqa: F401,F403from .backbones import *  # noqa: F401,F403

  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 14, in <module>
        from .bisa_transformer_ng import BisaSwinTransformerNgfrom .bisa_transformer_ng import BisaSwinTransformerNg

ModuleNotFoundError: No module named 'mmseg.models.backbones.bisa_transformer_ng'ModuleNotFoundError
:     No module named 'mmseg.models.backbones.bisa_transformer_ng'from .bisa_transformer_ng import BisaSwinTransformerNg

ModuleNotFoundError: No module named 'mmseg.models.backbones.bisa_transformer_ng'
    from .bisa_transformer_ng import BisaSwinTransformerNg
ModuleNotFoundError: No module named 'mmseg.models.backbones.bisa_transformer_ng'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 28207) of binary: /nethome/bdevnani3/flash1/envs/mmlab/bin/python
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
  time      : 2022-05-22_03:56:25
  host      : chomps.cc.gatech.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 28208)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2022-05-22_03:56:25
  host      : chomps.cc.gatech.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 28209)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2022-05-22_03:56:25
  host      : chomps.cc.gatech.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 28210)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2022-05-22_03:56:25
  host      : chomps.cc.gatech.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 28207)
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
2022-05-22 03:56:51,031 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: GeForce RTX 2080 Ti
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

2022-05-22 03:56:51,031 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:56:51,298 - mmseg - INFO - Config:
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
        apply_bidirectional_layer_norms=True,
        bidirectional_lambda_value=-100.0,
        lambda_learned=True,
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
work_dir = './work_dirs/bisa_norm_learned'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py(129)instantiate_generator_weights()
-> self.selection_lambda = nn.Parameter(torch.tensor(self.lambda_value, requires_grad=self.lambda_req_grad))
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py(129)instantiate_generator_weights()
-> self.selection_lambda = nn.Parameter(torch.tensor(self.lambda_value, requires_grad=self.lambda_req_grad))
(Pdb) /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py(129)instantiate_generator_weights()
-> self.selection_lambda = nn.Parameter(torch.tensor(self.lambda_value, requires_grad=self.lambda_req_grad))
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa.py(129)instantiate_generator_weights()
-> self.selection_lambda = nn.Parameter(torch.tensor(self.lambda_value, requires_grad=self.lambda_req_grad))
(Pdb) True
(Pdb) 