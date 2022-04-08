_base_ = [
    '../models/cascade_rcnn_pvtv1_pafpn.py',
    '../datasets/dataset_alb.py',
    '../schedules/schedule_1x.py',
    '../default_runtime.py'
]
pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth'
model = dict(
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth',
    backbone=dict(
        type='pvt_small',
        # pretrained=pretrained,
        style='pytorch'),
    neck=dict(
        type='PAFPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)