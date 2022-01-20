_base_ = [
    '../_base_/models/dpt_vit-b16.py', '../_base_/datasets/ade20k.py',
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
img_scale = tuple([int(i*1) for i in (850, 1200)])
crop_size = (896-64, 1280-64)
img_norm_cfg = dict(
    mean=[180, 180, 180], std=[70, 70, 70], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.1)),
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

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root='../mmdetection/datasets/manga',
        img_dir='data/downloaded',
        ann_dir='data/segment',
        split = ['data/segment/train.txt'],
        pipeline=train_pipeline,
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
    pretrained='./checkpoints/dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth'
    decode_head=dict(
        num_classes=len(classes)
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

eval_every = 1000 # 16000
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)

load_from = './checkpoints/dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth'
resume_from = './work_dirs/dpt_vit-b16_512x512_160k_manga/latest.pth'

# python tools/train.py configs/dpt/dpt_vit-b16_512x512_160k_manga.py
# python tools/print_config.py configs/dpt/dpt_vit-b16_512x512_160k_manga.py
# python tools/test.py configs/dpt/dpt_vit-b16_512x512_160k_manga.py work_dirs/dpt_vit-b16_512x512_160k_manga/iter_1000.pth --show-dir work_dirs/dpt_vit-b16_512x512_160k_manga/result_1000 --out test_result.pkl
