model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    },
    'dino_vit_base': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False,
    },
    'dinov2_vit_small': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False,
    },
    'dinov2_vit_base': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False,
    },
    'google/vit-base-patch16-224-in21k': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False,
    }
}