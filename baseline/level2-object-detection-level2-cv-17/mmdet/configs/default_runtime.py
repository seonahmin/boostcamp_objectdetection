checkpoint_config = dict(interval=2, max_keep_ckpts = 5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='NumClassCheckHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='object_detection_seonah',
                entity = 'mg_generation',
                # group = None,
                # config = None,
                # job_type = f'Fold{fold}',
                reinit = True
            ),
            )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook'),
                    # dict(type='MyHook',priority = 'LOWEST')
                    ]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
