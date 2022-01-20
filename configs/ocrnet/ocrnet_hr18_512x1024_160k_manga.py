_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
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
# img_scale = tuple([int(i*1) for i in (850, 1200)])
# crop_size = (1200, 850)
img_scale = tuple([int(i*1) for i in (425, 600)])
crop_size = (600, 425)
img_norm_cfg = dict(
    mean=[180, 180, 180], std=[70, 70, 70], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.7, 2.)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='Invert', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
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
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/segment/pages',
        ann_dir='data/segment/masks',
        split = ['data/segment/train.txt'],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/segment/pages',
        ann_dir='data/segment/masks',
        split = ['data/segment/test.txt'],
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/segment/pages',
        ann_dir='data/segment/masks',
        split = ['data/segment/test.txt'],
        pipeline=test_pipeline,
    )
)

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(decode_head=[
    dict(
        type='FCNHead',
        in_channels=[18, 36, 72, 144],
        channels=sum([18, 36, 72, 144]),
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=len(classes),
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    dict(
        type='OCRHead',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        channels=512,
        ocr_channels=256,
        dropout_ratio=-1,
        num_classes=len(classes),
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
])

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
eval_every = 1000
checkpoint_config = dict(by_epoch=False, interval=eval_every)
evaluation = dict(interval=eval_every, metric='mIoU', pre_eval=True)

# load_from = './checkpoints/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth'
load_from = './checkpoints/ocrnet_hr18_manga_iter_75000.pth'
# resume_from = './work_dirs/ocrnet_hr18_512x1024_160k_manga/latest.pth'

# python tools/train.py configs/ocrnet/ocrnet_hr18_512x1024_160k_manga.py
# python tools/print_config.py configs/ocrnet/ocrnet_hr18_512x1024_160k_manga.py
# python tools/browse_dataset.py configs/ocrnet/ocrnet_hr18_512x1024_160k_manga.py --output-dir work_dirs/ocrnet_hr18_512x1024_160k_manga/browse_dataset
# python tools/test.py configs/ocrnet/ocrnet_hr18_512x1024_160k_manga.py work_dirs/ocrnet_hr18_512x1024_160k_manga/iter_1000.pth --show-dir work_dirs/ocrnet_hr18_512x1024_160k_manga/result_1000 --out test_result.pkl --eval mIoU
