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
2022-05-22 03:01:26,904 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:01:26,904 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:01:27,218 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
Traceback (most recent call last):
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg

AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 28166) of binary: /nethome/bdevnani3/flash1/envs/mmlab/bin/python
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
  time      : 2022-05-22_03:01:31
  host      : tars.cc.gatech.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 28167)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2022-05-22_03:01:31
  host      : tars.cc.gatech.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 28168)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2022-05-22_03:01:31
  host      : tars.cc.gatech.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 28169)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2022-05-22_03:01:31
  host      : tars.cc.gatech.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 28166)
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
2022-05-22 03:02:25,043 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:02:25,043 - mmseg - INFO - Distributed training: True
2022-05-22 03:02:25,349 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 1056) of binary: /nethome/bdevnani3/flash1/envs/mmlab/bin/python
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
  time      : 2022-05-22_03:02:31
  host      : tars.cc.gatech.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 1057)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2022-05-22_03:02:31
  host      : tars.cc.gatech.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 1058)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2022-05-22_03:02:31
  host      : tars.cc.gatech.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 1059)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2022-05-22_03:02:31
  host      : tars.cc.gatech.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 1056)
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
2022-05-22 03:04:15,121 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609
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

2022-05-22 03:04:15,122 - mmseg - INFO - Distributed training: True
2022-05-22 03:04:16,674 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    Traceback (most recent call last):
nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
        return obj_cls(**args)return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_

  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    return tensor.fill_(val)    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights

AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
        self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main

  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    raise type(e)(f'{obj_cls.__name__}: {e}')    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply

AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 892) of binary: /nethome/bdevnani3/flash1/envs/mmlab/bin/python
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
  time      : 2022-05-22_03:04:27
  host      : eva.cc.gatech.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 893)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2022-05-22_03:04:27
  host      : eva.cc.gatech.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 894)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2022-05-22_03:04:27
  host      : eva.cc.gatech.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 895)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2022-05-22_03:04:27
  host      : eva.cc.gatech.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 892)
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
2022-05-22 03:14:28,805 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:14:28,806 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:14:29,132 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
Traceback (most recent call last):
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 39, in __init__
    self.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/segmentors/encoder_decoder.py", line 68, in init_weights
    self.backbone.init_weights(pretrained=pretrained)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 624, in init_weights
    self.apply(_init_weights)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 667, in apply
    module.apply(fn)
  [Previous line repeated 3 more times]
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py", line 668, in apply
    fn(self)
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 617, in _init_weights
    nn.init.constant_(m.bias, 0)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 192, in constant_
    return _no_grad_fill_(tensor, val)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/init.py", line 59, in _no_grad_fill_
    return tensor.fill_(val)
AttributeError: 'NoneType' object has no attribute 'fill_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 163, in <module>
    main()
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 130, in main
    model = build_segmentor(
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 66, in build_segmentor
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/builder.py", line 33, in build
    return build_from_cfg(cfg, registry, default_args)
  File "/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
AttributeError: EncoderDecoder: 'NoneType' object has no attribute 'fill_'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 29979) of binary: /nethome/bdevnani3/flash1/envs/mmlab/bin/python
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
  time      : 2022-05-22_03:14:35
  host      : tars.cc.gatech.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 29980)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2022-05-22_03:14:35
  host      : tars.cc.gatech.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 30050)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2022-05-22_03:14:35
  host      : tars.cc.gatech.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 30075)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2022-05-22_03:14:35
  host      : tars.cc.gatech.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 29979)
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
2022-05-22 03:16:03,782 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:16:03,783 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:16:04,143 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(612)init_weights()
-> def _init_weights(m):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(612)init_weights()
-> def _init_weights(m):> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(612)init_weights()
-> def _init_weights(m):

(Pdb) (Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(612)init_weights()
-> def _init_weights(m):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(621)init_weights()
-> if isinstance(pretrained, str):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(621)init_weights()
-> if isinstance(pretrained, str):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(621)init_weights()
-> if isinstance(pretrained, str):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(621)init_weights()
-> if isinstance(pretrained, str):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(624)init_weights()
-> elif pretrained is None:
(Pdb) /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:19:38,894 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:19:38,895 - mmseg - INFO - Distributed training: True
2022-05-22 03:19:39,214 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(612)init_weights()
-> def _init_weights(m):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(621)init_weights()
-> if isinstance(pretrained, str):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(624)init_weights()
-> elif pretrained is None:
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(625)init_weights()
-> self.apply(_init_weights)
(Pdb) /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
Traceback (most recent call last):
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/tools/train.py", line 13, in <module>
    from mmseg.apis import set_random_seed, train_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/__init__.py", line 1, in <module>
    from .inference import inference_segmentor, init_segmentor, show_result_pyplot
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/apis/inference.py", line 8, in <module>
    from mmseg.models import build_segmentor
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/__init__.py", line 1, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/__init__.py", line 13, in <module>
    from .bisa_transformer import BisaSwinTransformer
  File "/coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py", line 614
    if isinstance(m, nn.Linear):
                                ^
IndentationError: unindent does not match any outer indentation level
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 37419) of binary: /nethome/bdevnani3/flash1/envs/mmlab/bin/python
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
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2022-05-22_03:20:29
  host      : tars.cc.gatech.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 37419)
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
2022-05-22 03:20:43,191 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:20:43,191 - mmseg - INFO - Distributed training: True
2022-05-22 03:20:43,508 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(614)_init_weights()
-> if isinstance(m, nn.Linear):
(Pdb) Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()
-> elif isinstance(m, nn.LayerNorm):
(Pdb) --Return--
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()->None
-> elif isinstance(m, nn.LayerNorm):
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()
-> return self
(Pdb) --Return--
> /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()->Conv2d(3, 96,...stride=(4, 4))
-> return self
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(666)apply()
-> for module in self.children():
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(667)apply()
-> module.apply(fn)
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(614)_init_weights()
-> if isinstance(m, nn.Linear):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()
-> elif isinstance(m, nn.LayerNorm):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(619)_init_weights()
-> nn.init.constant_(m.bias, 0)
(Pdb) LayerNorm((96,), eps=1e-05, elementwise_affine=True)
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(620)_init_weights()
-> nn.init.constant_(m.weight, 1.0)
(Pdb) LayerNorm((96,), eps=1e-05, elementwise_affine=True)
(Pdb) --Return--
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(620)_init_weights()->None
-> nn.init.constant_(m.weight, 1.0)
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()
-> return self
(Pdb) --Return--
> /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()->LayerNorm((96...e_affine=True)
-> return self
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(666)apply()
-> for module in self.children():
(Pdb) Internal StopIteration
> /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(666)apply()
-> for module in self.children():
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(668)apply()
-> fn(self)
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(614)_init_weights()
-> if isinstance(m, nn.Linear):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()
-> elif isinstance(m, nn.LayerNorm):
(Pdb) --Return--
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()->None
-> elif isinstance(m, nn.LayerNorm):
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()
-> return self
(Pdb) --Return--
> /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()->PatchEmbed(
 ...affine=True)
)
-> return self
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(666)apply()
-> for module in self.children():
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(667)apply()
-> module.apply(fn)
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(614)_init_weights()
-> if isinstance(m, nn.Linear):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()
-> elif isinstance(m, nn.LayerNorm):
(Pdb) --Return--
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()->None
-> elif isinstance(m, nn.LayerNorm):
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()
-> return self
(Pdb) --Return--
> /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()->Dropout(p=0.0, inplace=False)
-> return self
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(666)apply()
-> for module in self.children():
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(667)apply()
-> module.apply(fn)
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(614)_init_weights()
-> if isinstance(m, nn.Linear):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(618)_init_weights()
-> elif isinstance(m, nn.LayerNorm):
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(619)_init_weights()
-> nn.init.constant_(m.bias, 0)
(Pdb) > /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(620)_init_weights()
-> nn.init.constant_(m.weight, 1.0)
(Pdb) --Return--
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(620)_init_weights()->None
-> nn.init.constant_(m.weight, 1.0)
(Pdb) > /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()
-> return self
(Pdb) --Return--
> /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/nn/modules/module.py(669)apply()->LayerNorm((96...e_affine=True)
-> return self
(Pdb) /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:22:47,392 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:22:47,392 - mmseg - INFO - Distributed training: True
2022-05-22 03:22:47,715 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
> /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/mmseg/models/backbones/bisa_transformer.py(614)_init_weights()
-> if isinstance(m, nn.Linear):
(Pdb) /nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:23:19,865 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:23:19,866 - mmseg - INFO - Distributed training: True
2022-05-22 03:23:20,189 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:23:22,107 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 03:23:23,211 - mmseg - INFO - Loaded 20210 images
2022-05-22 03:23:30,147 - mmseg - INFO - Loaded 2000 images
2022-05-22 03:23:30,148 - mmseg - INFO - Start running, host: bdevnani3@tars, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_0-5
2022-05-22 03:23:30,148 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 03:23:30,148 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 03:23:30,149 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_0-5 by HardDiskBackend.
2022-05-22 03:23:43,742 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 03:23:45,652 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
2022-05-22 03:24:12,062 - mmseg - INFO - Iter [50/320000]	lr: 9.799e-07, eta: 2 days, 7:07:59, time: 0.620, data_time: 0.006, memory: 8043, decode.loss_seg: 4.3128, decode.acc_seg: 0.9691, aux.loss_seg: 1.7308, aux.acc_seg: 0.7163, loss: 6.0436
2022-05-22 03:24:39,130 - mmseg - INFO - Iter [100/320000]	lr: 1.979e-06, eta: 2 days, 3:36:52, time: 0.541, data_time: 0.003, memory: 8043, decode.loss_seg: 4.2199, decode.acc_seg: 0.8448, aux.loss_seg: 1.6891, aux.acc_seg: 0.2987, loss: 5.9090
2022-05-22 03:25:06,274 - mmseg - INFO - Iter [150/320000]	lr: 2.979e-06, eta: 2 days, 2:28:54, time: 0.543, data_time: 0.003, memory: 8043, decode.loss_seg: 4.0411, decode.acc_seg: 0.9860, aux.loss_seg: 1.6240, aux.acc_seg: 0.4077, loss: 5.6650
2022-05-22 03:25:33,521 - mmseg - INFO - Iter [200/320000]	lr: 3.978e-06, eta: 2 days, 1:57:26, time: 0.545, data_time: 0.003, memory: 8043, decode.loss_seg: 3.9953, decode.acc_seg: 4.6907, aux.loss_seg: 1.6209, aux.acc_seg: 0.3190, loss: 5.6163
2022-05-22 03:26:00,659 - mmseg - INFO - Iter [250/320000]	lr: 4.976e-06, eta: 2 days, 1:36:03, time: 0.543, data_time: 0.003, memory: 8043, decode.loss_seg: 3.9197, decode.acc_seg: 8.2769, aux.loss_seg: 1.5973, aux.acc_seg: 0.6153, loss: 5.5170
2022-05-22 03:26:27,777 - mmseg - INFO - Iter [300/320000]	lr: 5.974e-06, eta: 2 days, 1:21:17, time: 0.542, data_time: 0.003, memory: 8043, decode.loss_seg: 3.9107, decode.acc_seg: 9.7699, aux.loss_seg: 1.6106, aux.acc_seg: 1.3841, loss: 5.5213
2022-05-22 03:26:54,743 - mmseg - INFO - Iter [350/320000]	lr: 6.972e-06, eta: 2 days, 1:08:18, time: 0.539, data_time: 0.003, memory: 8043, decode.loss_seg: 3.8215, decode.acc_seg: 11.8311, aux.loss_seg: 1.5851, aux.acc_seg: 4.8667, loss: 5.4066
2022-05-22 03:27:21,658 - mmseg - INFO - Iter [400/320000]	lr: 7.970e-06, eta: 2 days, 0:57:46, time: 0.538, data_time: 0.003, memory: 8043, decode.loss_seg: 3.9649, decode.acc_seg: 11.8675, aux.loss_seg: 1.6586, aux.acc_seg: 6.4038, loss: 5.6234
2022-05-22 03:27:48,484 - mmseg - INFO - Iter [450/320000]	lr: 8.967e-06, eta: 2 days, 0:48:25, time: 0.537, data_time: 0.003, memory: 8043, decode.loss_seg: 3.7196, decode.acc_seg: 13.7834, aux.loss_seg: 1.5768, aux.acc_seg: 9.9538, loss: 5.2964
2022-05-22 03:28:15,188 - mmseg - INFO - Iter [500/320000]	lr: 9.964e-06, eta: 2 days, 0:39:34, time: 0.534, data_time: 0.003, memory: 8043, decode.loss_seg: 3.6619, decode.acc_seg: 14.0874, aux.loss_seg: 1.5536, aux.acc_seg: 9.6576, loss: 5.2154
2022-05-22 03:28:41,870 - mmseg - INFO - Iter [550/320000]	lr: 1.096e-05, eta: 2 days, 0:32:01, time: 0.534, data_time: 0.003, memory: 8043, decode.loss_seg: 3.8606, decode.acc_seg: 13.1378, aux.loss_seg: 1.6425, aux.acc_seg: 9.2290, loss: 5.5031
2022-05-22 03:29:08,504 - mmseg - INFO - Iter [600/320000]	lr: 1.196e-05, eta: 2 days, 0:25:14, time: 0.533, data_time: 0.003, memory: 8043, decode.loss_seg: 3.6967, decode.acc_seg: 14.0930, aux.loss_seg: 1.5855, aux.acc_seg: 10.5696, loss: 5.2822
2022-05-22 03:29:35,145 - mmseg - INFO - Iter [650/320000]	lr: 1.295e-05, eta: 2 days, 0:19:29, time: 0.533, data_time: 0.003, memory: 8043, decode.loss_seg: 3.4133, decode.acc_seg: 15.1605, aux.loss_seg: 1.4810, aux.acc_seg: 11.5898, loss: 4.8942
2022-05-22 03:30:01,742 - mmseg - INFO - Iter [700/320000]	lr: 1.395e-05, eta: 2 days, 0:14:09, time: 0.532, data_time: 0.003, memory: 8043, decode.loss_seg: 3.3176, decode.acc_seg: 17.6425, aux.loss_seg: 1.4537, aux.acc_seg: 13.7406, loss: 4.7712
2022-05-22 03:30:28,349 - mmseg - INFO - Iter [750/320000]	lr: 1.494e-05, eta: 2 days, 0:09:32, time: 0.532, data_time: 0.003, memory: 8043, decode.loss_seg: 3.5278, decode.acc_seg: 16.0207, aux.loss_seg: 1.5286, aux.acc_seg: 10.9185, loss: 5.0563
2022-05-22 03:30:54,983 - mmseg - INFO - Iter [800/320000]	lr: 1.594e-05, eta: 2 days, 0:05:38, time: 0.533, data_time: 0.003, memory: 8043, decode.loss_seg: 3.5225, decode.acc_seg: 14.0421, aux.loss_seg: 1.5198, aux.acc_seg: 10.7289, loss: 5.0423
2022-05-22 03:31:21,564 - mmseg - INFO - Iter [850/320000]	lr: 1.693e-05, eta: 2 days, 0:01:48, time: 0.532, data_time: 0.003, memory: 8043, decode.loss_seg: 3.3340, decode.acc_seg: 16.5640, aux.loss_seg: 1.4736, aux.acc_seg: 12.3541, loss: 4.8076
2022-05-22 03:31:48,111 - mmseg - INFO - Iter [900/320000]	lr: 1.793e-05, eta: 1 day, 23:58:08, time: 0.531, data_time: 0.003, memory: 8043, decode.loss_seg: 3.2353, decode.acc_seg: 20.5252, aux.loss_seg: 1.4497, aux.acc_seg: 13.9946, loss: 4.6850
2022-05-22 03:32:14,534 - mmseg - INFO - Iter [950/320000]	lr: 1.892e-05, eta: 1 day, 23:54:08, time: 0.528, data_time: 0.003, memory: 8043, decode.loss_seg: 3.3170, decode.acc_seg: 18.0655, aux.loss_seg: 1.4766, aux.acc_seg: 14.0133, loss: 4.7936
2022-05-22 03:32:41,040 - mmseg - INFO - Exp name: bisa_norm_0-5.py
2022-05-22 03:32:41,041 - mmseg - INFO - Iter [1000/320000]	lr: 1.992e-05, eta: 1 day, 23:50:55, time: 0.530, data_time: 0.003, memory: 8043, decode.loss_seg: 3.3749, decode.acc_seg: 21.7870, aux.loss_seg: 1.5176, aux.acc_seg: 15.7509, loss: 4.8925
2022-05-22 03:33:07,513 - mmseg - INFO - Iter [1050/320000]	lr: 2.091e-05, eta: 1 day, 23:47:48, time: 0.529, data_time: 0.003, memory: 8043, decode.loss_seg: 3.1903, decode.acc_seg: 18.2625, aux.loss_seg: 1.4356, aux.acc_seg: 13.9096, loss: 4.6259
2022-05-22 03:33:34,027 - mmseg - INFO - Iter [1100/320000]	lr: 2.190e-05, eta: 1 day, 23:45:07, time: 0.530, data_time: 0.003, memory: 8043, decode.loss_seg: 3.1687, decode.acc_seg: 16.5459, aux.loss_seg: 1.4191, aux.acc_seg: 12.2500, loss: 4.5878
2022-05-22 03:34:00,445 - mmseg - INFO - Iter [1150/320000]	lr: 2.290e-05, eta: 1 day, 23:42:12, time: 0.528, data_time: 0.003, memory: 8043, decode.loss_seg: 3.2343, decode.acc_seg: 20.5424, aux.loss_seg: 1.4654, aux.acc_seg: 16.4665, loss: 4.6997
2022-05-22 03:34:26,735 - mmseg - INFO - Iter [1200/320000]	lr: 2.389e-05, eta: 1 day, 23:38:55, time: 0.526, data_time: 0.003, memory: 8043, decode.loss_seg: 3.2789, decode.acc_seg: 18.0121, aux.loss_seg: 1.4702, aux.acc_seg: 14.0091, loss: 4.7491
2022-05-22 03:34:53,094 - mmseg - INFO - Iter [1250/320000]	lr: 2.488e-05, eta: 1 day, 23:36:09, time: 0.527, data_time: 0.003, memory: 8043, decode.loss_seg: 3.0944, decode.acc_seg: 20.1419, aux.loss_seg: 1.4003, aux.acc_seg: 16.6171, loss: 4.4947
2022-05-22 03:35:19,318 - mmseg - INFO - Iter [1300/320000]	lr: 2.587e-05, eta: 1 day, 23:33:01, time: 0.524, data_time: 0.003, memory: 8043, decode.loss_seg: 3.0257, decode.acc_seg: 21.5683, aux.loss_seg: 1.3943, aux.acc_seg: 18.5025, loss: 4.4200
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:35:30,854 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:35:30,854 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:35:31,161 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:35:34,017 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 03:35:37,029 - mmseg - INFO - Loaded 20210 images
2022-05-22 03:35:42,909 - mmseg - INFO - Loaded 2000 images
2022-05-22 03:35:42,910 - mmseg - INFO - Start running, host: bdevnani3@tars, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_0-5
2022-05-22 03:35:42,911 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 03:35:42,911 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 03:35:42,911 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_0-5 by HardDiskBackend.
2022-05-22 03:35:56,733 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 03:35:58,761 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
2022-05-22 03:36:27,509 - mmseg - INFO - Iter [50/320000]	lr: 9.799e-07, eta: 2 days, 11:43:45, time: 0.672, data_time: 0.012, memory: 8044, decode.loss_seg: 4.0884, decode.acc_seg: 0.9661, aux.loss_seg: 1.6496, aux.acc_seg: 0.2025, loss: 5.7380
2022-05-22 03:36:56,967 - mmseg - INFO - Iter [100/320000]	lr: 1.979e-06, eta: 2 days, 8:02:07, time: 0.589, data_time: 0.005, memory: 8044, decode.loss_seg: 4.0233, decode.acc_seg: 1.5824, aux.loss_seg: 1.6316, aux.acc_seg: 0.2241, loss: 5.6549
2022-05-22 03:37:26,382 - mmseg - INFO - Iter [150/320000]	lr: 2.979e-06, eta: 2 days, 6:46:24, time: 0.588, data_time: 0.004, memory: 8044, decode.loss_seg: 3.9599, decode.acc_seg: 5.6192, aux.loss_seg: 1.6184, aux.acc_seg: 0.3695, loss: 5.5783
2022-05-22 03:37:55,837 - mmseg - INFO - Iter [200/320000]	lr: 3.978e-06, eta: 2 days, 6:09:23, time: 0.589, data_time: 0.004, memory: 8044, decode.loss_seg: 3.8635, decode.acc_seg: 10.8892, aux.loss_seg: 1.5948, aux.acc_seg: 0.8091, loss: 5.4583
2022-05-22 03:38:25,341 - mmseg - INFO - Iter [250/320000]	lr: 4.976e-06, eta: 2 days, 5:48:00, time: 0.590, data_time: 0.004, memory: 8044, decode.loss_seg: 3.8514, decode.acc_seg: 14.5503, aux.loss_seg: 1.6163, aux.acc_seg: 2.5495, loss: 5.4677
2022-05-22 03:38:54,657 - mmseg - INFO - Iter [300/320000]	lr: 5.974e-06, eta: 2 days, 5:30:16, time: 0.586, data_time: 0.003, memory: 8044, decode.loss_seg: 3.7623, decode.acc_seg: 13.7149, aux.loss_seg: 1.5826, aux.acc_seg: 5.8750, loss: 5.3449
2022-05-22 03:39:23,857 - mmseg - INFO - Iter [350/320000]	lr: 6.972e-06, eta: 2 days, 5:15:42, time: 0.584, data_time: 0.003, memory: 8044, decode.loss_seg: 3.7670, decode.acc_seg: 17.3212, aux.loss_seg: 1.6131, aux.acc_seg: 10.2030, loss: 5.3800
2022-05-22 03:39:53,012 - mmseg - INFO - Iter [400/320000]	lr: 7.970e-06, eta: 2 days, 5:04:01, time: 0.583, data_time: 0.004, memory: 8044, decode.loss_seg: 3.6987, decode.acc_seg: 17.5675, aux.loss_seg: 1.5899, aux.acc_seg: 12.3158, loss: 5.2886
2022-05-22 03:40:21,997 - mmseg - INFO - Iter [450/320000]	lr: 8.967e-06, eta: 2 days, 4:52:50, time: 0.580, data_time: 0.003, memory: 8044, decode.loss_seg: 3.5231, decode.acc_seg: 18.7731, aux.loss_seg: 1.5381, aux.acc_seg: 14.2241, loss: 5.0612
2022-05-22 03:40:50,959 - mmseg - INFO - Iter [500/320000]	lr: 9.964e-06, eta: 2 days, 4:43:33, time: 0.579, data_time: 0.003, memory: 8044, decode.loss_seg: 3.4739, decode.acc_seg: 19.2273, aux.loss_seg: 1.5296, aux.acc_seg: 15.3892, loss: 5.0035
2022-05-22 03:41:19,986 - mmseg - INFO - Iter [550/320000]	lr: 1.096e-05, eta: 2 days, 4:36:30, time: 0.581, data_time: 0.003, memory: 8044, decode.loss_seg: 3.5366, decode.acc_seg: 18.8980, aux.loss_seg: 1.5456, aux.acc_seg: 14.5884, loss: 5.0822
2022-05-22 03:41:49,022 - mmseg - INFO - Iter [600/320000]	lr: 1.196e-05, eta: 2 days, 4:30:36, time: 0.581, data_time: 0.003, memory: 8044, decode.loss_seg: 3.4258, decode.acc_seg: 21.0419, aux.loss_seg: 1.5169, aux.acc_seg: 16.8405, loss: 4.9427
2022-05-22 03:42:18,059 - mmseg - INFO - Iter [650/320000]	lr: 1.295e-05, eta: 2 days, 4:25:34, time: 0.581, data_time: 0.004, memory: 8044, decode.loss_seg: 3.4455, decode.acc_seg: 20.5668, aux.loss_seg: 1.5293, aux.acc_seg: 17.1364, loss: 4.9748
2022-05-22 03:42:47,045 - mmseg - INFO - Iter [700/320000]	lr: 1.395e-05, eta: 2 days, 4:20:47, time: 0.580, data_time: 0.003, memory: 8044, decode.loss_seg: 3.2162, decode.acc_seg: 21.3200, aux.loss_seg: 1.4552, aux.acc_seg: 17.6514, loss: 4.6714
2022-05-22 03:43:15,986 - mmseg - INFO - Iter [750/320000]	lr: 1.494e-05, eta: 2 days, 4:16:15, time: 0.579, data_time: 0.003, memory: 8044, decode.loss_seg: 3.2051, decode.acc_seg: 21.3620, aux.loss_seg: 1.4483, aux.acc_seg: 18.5450, loss: 4.6533
2022-05-22 03:43:44,984 - mmseg - INFO - Iter [800/320000]	lr: 1.594e-05, eta: 2 days, 4:12:32, time: 0.580, data_time: 0.004, memory: 8044, decode.loss_seg: 3.3012, decode.acc_seg: 22.1414, aux.loss_seg: 1.4996, aux.acc_seg: 19.3519, loss: 4.8009
2022-05-22 03:44:13,846 - mmseg - INFO - Iter [850/320000]	lr: 1.693e-05, eta: 2 days, 4:08:29, time: 0.577, data_time: 0.004, memory: 8044, decode.loss_seg: 3.1156, decode.acc_seg: 22.9099, aux.loss_seg: 1.4310, aux.acc_seg: 19.4449, loss: 4.5466
2022-05-22 03:44:42,750 - mmseg - INFO - Iter [900/320000]	lr: 1.793e-05, eta: 2 days, 4:05:01, time: 0.578, data_time: 0.003, memory: 8044, decode.loss_seg: 3.1139, decode.acc_seg: 22.4102, aux.loss_seg: 1.4277, aux.acc_seg: 19.1763, loss: 4.5415
2022-05-22 03:45:11,607 - mmseg - INFO - Iter [950/320000]	lr: 1.892e-05, eta: 2 days, 4:01:36, time: 0.577, data_time: 0.003, memory: 8044, decode.loss_seg: 3.0960, decode.acc_seg: 23.8959, aux.loss_seg: 1.4342, aux.acc_seg: 20.9891, loss: 4.5302
2022-05-22 03:45:40,431 - mmseg - INFO - Exp name: bisa_norm_0-5.py
2022-05-22 03:45:40,432 - mmseg - INFO - Iter [1000/320000]	lr: 1.992e-05, eta: 2 days, 3:58:18, time: 0.577, data_time: 0.004, memory: 8044, decode.loss_seg: 2.9889, decode.acc_seg: 23.4666, aux.loss_seg: 1.4002, aux.acc_seg: 19.8825, loss: 4.3891
2022-05-22 03:46:09,187 - mmseg - INFO - Iter [1050/320000]	lr: 2.091e-05, eta: 2 days, 3:54:55, time: 0.575, data_time: 0.004, memory: 8044, decode.loss_seg: 2.8933, decode.acc_seg: 22.7196, aux.loss_seg: 1.3528, aux.acc_seg: 19.7682, loss: 4.2462
2022-05-22 03:46:37,961 - mmseg - INFO - Iter [1100/320000]	lr: 2.190e-05, eta: 2 days, 3:51:54, time: 0.575, data_time: 0.004, memory: 8044, decode.loss_seg: 2.9634, decode.acc_seg: 24.5245, aux.loss_seg: 1.3901, aux.acc_seg: 21.1612, loss: 4.3535
2022-05-22 03:47:06,823 - mmseg - INFO - Iter [1150/320000]	lr: 2.290e-05, eta: 2 days, 3:49:30, time: 0.577, data_time: 0.004, memory: 8044, decode.loss_seg: 2.8360, decode.acc_seg: 24.0188, aux.loss_seg: 1.3361, aux.acc_seg: 20.9264, loss: 4.1721
2022-05-22 03:47:35,586 - mmseg - INFO - Iter [1200/320000]	lr: 2.389e-05, eta: 2 days, 3:46:49, time: 0.575, data_time: 0.004, memory: 8044, decode.loss_seg: 2.7554, decode.acc_seg: 25.9195, aux.loss_seg: 1.3159, aux.acc_seg: 22.2061, loss: 4.0713
2022-05-22 03:48:04,329 - mmseg - INFO - Iter [1250/320000]	lr: 2.488e-05, eta: 2 days, 3:44:14, time: 0.575, data_time: 0.003, memory: 8044, decode.loss_seg: 2.8482, decode.acc_seg: 25.4699, aux.loss_seg: 1.3448, aux.acc_seg: 21.3648, loss: 4.1931
2022-05-22 03:48:33,208 - mmseg - INFO - Iter [1300/320000]	lr: 2.587e-05, eta: 2 days, 3:42:20, time: 0.577, data_time: 0.004, memory: 8044, decode.loss_seg: 2.8413, decode.acc_seg: 25.6586, aux.loss_seg: 1.3395, aux.acc_seg: 21.3692, loss: 4.1808
2022-05-22 03:49:01,948 - mmseg - INFO - Iter [1350/320000]	lr: 2.687e-05, eta: 2 days, 3:40:04, time: 0.575, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6597, decode.acc_seg: 23.9083, aux.loss_seg: 1.2631, aux.acc_seg: 20.9940, loss: 3.9228
2022-05-22 03:49:30,762 - mmseg - INFO - Iter [1400/320000]	lr: 2.786e-05, eta: 2 days, 3:38:09, time: 0.576, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6008, decode.acc_seg: 24.6178, aux.loss_seg: 1.2423, aux.acc_seg: 20.2919, loss: 3.8431
2022-05-22 03:49:59,546 - mmseg - INFO - Iter [1450/320000]	lr: 2.885e-05, eta: 2 days, 3:36:15, time: 0.576, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6430, decode.acc_seg: 25.4684, aux.loss_seg: 1.2587, aux.acc_seg: 21.7704, loss: 3.9017
2022-05-22 03:50:28,366 - mmseg - INFO - Iter [1500/320000]	lr: 2.984e-05, eta: 2 days, 3:34:33, time: 0.576, data_time: 0.004, memory: 8044, decode.loss_seg: 2.6147, decode.acc_seg: 25.1595, aux.loss_seg: 1.2393, aux.acc_seg: 21.5011, loss: 3.8541
2022-05-22 03:50:57,185 - mmseg - INFO - Iter [1550/320000]	lr: 3.083e-05, eta: 2 days, 3:32:57, time: 0.576, data_time: 0.003, memory: 8044, decode.loss_seg: 2.5774, decode.acc_seg: 26.4267, aux.loss_seg: 1.2190, aux.acc_seg: 22.8724, loss: 3.7964
2022-05-22 03:51:25,825 - mmseg - INFO - Iter [1600/320000]	lr: 3.182e-05, eta: 2 days, 3:30:48, time: 0.573, data_time: 0.003, memory: 8044, decode.loss_seg: 2.4790, decode.acc_seg: 26.2815, aux.loss_seg: 1.1692, aux.acc_seg: 22.6024, loss: 3.6482
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
2022-05-22 03:58:21,777 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.9.12 | packaged by conda-forge | (main, Mar 24 2022, 23:22:55) [GCC 10.3.0]
CUDA available: True
GPU 0,1,2,3: TITAN Xp
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609
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

2022-05-22 03:58:21,777 - mmseg - INFO - Distributed training: True
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:58:22,094 - mmseg - INFO - Config:
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
work_dir = './work_dirs/bisa_norm_0-5'
gpu_ids = range(0, 1)

/nethome/bdevnani3/flash1/envs/mmlab/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646756402876/work/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-22 03:58:24,895 - mmseg - INFO - EncoderDecoder(
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
2022-05-22 03:58:25,447 - mmseg - INFO - Loaded 20210 images
2022-05-22 03:58:31,224 - mmseg - INFO - Loaded 2000 images
2022-05-22 03:58:31,224 - mmseg - INFO - Start running, host: bdevnani3@tars, work_dir: /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_0-5
2022-05-22 03:58:31,224 - mmseg - INFO - Hooks will be executed in the following order:
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
2022-05-22 03:58:31,224 - mmseg - INFO - workflow: [('train', 1)], max: 320000 iters
2022-05-22 03:58:31,225 - mmseg - INFO - Checkpoints will be saved to /coc/testnvme/bdevnani3/Swin-Transformer-Semantic-Segmentation/work_dirs/bisa_norm_0-5 by HardDiskBackend.
2022-05-22 03:58:45,008 - mmseg - WARNING - GradientCumulativeOptimizerHook may slightly decrease performance if the model has BatchNorm layers.
2022-05-22 03:58:46,987 - mmcv - INFO - Reducer buckets have been rebuilt in this iteration.
