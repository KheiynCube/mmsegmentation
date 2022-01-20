_base_ = ['./segformer_mit-b0_512x512_160k_ade20k.py']

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
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.7, 1.4)),
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
    samples_per_gpu=4,
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

model = dict(
    pretrained='pretrain/mit_b2.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=len(classes)
    )
)

eval_every = 1000 # 16000
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=eval_every)
evaluation = dict(interval=eval_every, metric='mIoU', pre_eval=True)


load_from = './checkpoints/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth'
resume_from = './work_dirs/segformer_mit-b2_600x425_manga/latest.pth'

# python tools/train.py configs/segformer/segformer_mit-b2_600x425_manga.py
# python tools/print_config.py configs/segformer/segformer_mit-b2_600x425_manga.py
# python tools/browse_dataset.py configs/segformer/segformer_mit-b2_600x425_manga.py --output-dir work_dirs/segformer_mit-b2_600x425_manga/browse_dataset
# python tools/test.py configs/segformer/segformer_mit-b2_600x425_manga.py work_dirs/segformer_mit-b2_600x425_manga/iter_1000.pth --show-dir work_dirs/segformer_mit-b2_600x425_manga/result_1000 --out test_result.pkl --eval mIoU
