_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/hrf.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

# 1. dataset settings
# Modify dataset related settings

classes = (
    'background', 
    'balloon', 
    'text_format',
    'text_effect',
)

dataset_type='MangaDataset'
img_scale = tuple([int(i*1) for i in (850, 1200)])
img_norm_cfg = dict(
    mean=[180, 180, 180], std=[70, 70, 70], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='Invert', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
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
            dict(type='Resize', img_scale=img_scale, ratio_range=True),
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
        type='RepeatDataset',
        times=40,
        dataset=dict(
            type=dataset_type,
            data_root='../mmdetection/datasets/manga',
            img_dir='data/downloaded',
            ann_dir='data/segment',
            split = ['data/segment/train.txt'],
            pipeline=train_pipeline,
        )
    ),
    val=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/downloaded',
        ann_dir='data/segment',
        split = ['data/segment/test.txt'],
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/downloaded',
        ann_dir='data/segment',
        split = ['data/segment/test.txt'],
        pipeline=test_pipeline,
    )
)

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation

model = dict(
    decode_head=(dict(num_classes=len(classes))),
    auxiliary_head=(dict(num_classes=len(classes))),
    test_cfg=dict(mode='whole'))

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=400)
evaluation = dict(interval=400, metric='mDice', pre_eval=True)

load_from = './checkpoints/fcn_unet_s5-d16_64x64_40k_drive_20201223_191051-5daf6d3b.pth'
# resume_from = './work_dirs/mask_rcnn_r50_fpn_1x_cell/latest.pth'


# python tools/train.py configs/unet/fcn_unet_s5-d16_256x256_manga.py
# python tools/print_config.py configs/unet/fcn_unet_s5-d16_256x256_manga.py
# python tools/test.py configs/unet/fcn_unet_s5-d16_256x256_manga.py work_dirs/fcn_unet_s5-d16_256x256_manga/iter_400.pth --show-dir work_dirs/fcn_unet_s5-d16_256x256_manga/result_400

