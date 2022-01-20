_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

# 1. dataset settings
# Modify dataset related settings

dataset_type='MangaDataset'
img_scale = tuple([int(i*1) for i in (850, 1200)])
img_norm_cfg = dict(
    mean=[180, 180, 180], std=[70, 70, 70], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=img_scale, keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/downloaded',
        ann_dir='data/masks',
        split = ['train.txt'],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/downloaded',
        ann_dir='data/masks',
        split = ['test.txt'],
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/downloaded',
        ann_dir='data/masks',
        split = ['test.txt'],
        pipeline=test_pipeline,
    )
)

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
optimizer = dict(lr=0.02)
lr_config = dict(min_lr=2e-4)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
load_from = './checkpoints/ocrnet_r101-d8_512x1024_80k_b16_cityscapes_20200723_192421-78688424.pth'
load_from = './checkpoints/ocrnet_r101_text_5500.pth'

# python tools/train.py configs/ocrnet/ocrnet_r101-d8_512x1024_80k_b16_manga.py
# python tools/print_config.py configs/ocrnet/ocrnet_r101-d8_512x1024_80k_b16_manga.py
# python tools/test.py configs/ocrnet/ocrnet_r101-d8_512x1024_80k_b16_manga.py work_dirs/ocrnet_r101-d8_512x1024_80k_b16_manga/iter_500.pth --show-dir work_dirs/ocrnet_r101-d8_512x1024_80k_b16_manga/result --out test_result.pkl