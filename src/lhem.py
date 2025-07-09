"""Lightweight Hybrid Embedding Module (LHEM)"""
# Implements DistilBERT and EfficientNet hybrid embedding

def process_input(input_data, image_model_name='EfficientNetB0'):
    """
    Process input using LHEM.
    Args:
        input_data (dict): {'text': str, 'image': np.ndarray or None}
        image_model_name (str): Name of Keras image model to use (default: 'EfficientNetB0')
    Returns:
        dict: {'text_embedding': np.ndarray, 'image_embedding': np.ndarray or None}
    """
    if (not input_data.get('text')) and (input_data.get('image') is None):
        raise ValueError('At least one of text or image input must be provided.')
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    import numpy as np
    # Import all major Keras image models
    from tensorflow.keras.applications import (
        EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
        ResNet50, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, Xception, MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
        DenseNet121, DenseNet169, DenseNet201, NASNetMobile, NASNetLarge, VGG16, VGG19
    )
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnet_preprocess
    from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
    from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
    from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
    from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
    from tensorflow.keras.preprocessing import image as keras_image

    # Text embedding
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    inputs = tokenizer(input_data['text'], return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Image embedding (optional)
    img_embed = None
    if input_data.get('image') is not None:
        img = input_data['image']
        if img.shape[-1] != 3:
            raise ValueError('Image must have 3 channels (RGB)')
        img_resized = keras_image.smart_resize(img, (224, 224))
        # Model selection and preprocessing
        model_map = {
            'EfficientNetB0': (EfficientNetB0, efficientnet_preprocess),
            'EfficientNetB1': (EfficientNetB1, efficientnet_preprocess),
            'EfficientNetB2': (EfficientNetB2, efficientnet_preprocess),
            'EfficientNetB3': (EfficientNetB3, efficientnet_preprocess),
            'EfficientNetB4': (EfficientNetB4, efficientnet_preprocess),
            'EfficientNetB5': (EfficientNetB5, efficientnet_preprocess),
            'EfficientNetB6': (EfficientNetB6, efficientnet_preprocess),
            'EfficientNetB7': (EfficientNetB7, efficientnet_preprocess),
            'ResNet50': (ResNet50, resnet_preprocess),
            'ResNet101': (ResNet101, resnet_preprocess),
            'ResNet152': (ResNet152, resnet_preprocess),
            'InceptionV3': (InceptionV3, inception_preprocess),
            'InceptionResNetV2': (InceptionResNetV2, inceptionresnet_preprocess),
            'Xception': (Xception, xception_preprocess),
            'MobileNet': (MobileNet, mobilenet_preprocess),
            'MobileNetV2': (MobileNetV2, mobilenetv2_preprocess),
            'MobileNetV3Small': (MobileNetV3Small, mobilenetv3_preprocess),
            'MobileNetV3Large': (MobileNetV3Large, mobilenetv3_preprocess),
            'DenseNet121': (DenseNet121, densenet_preprocess),
            'DenseNet169': (DenseNet169, densenet_preprocess),
            'DenseNet201': (DenseNet201, densenet_preprocess),
            'NASNetMobile': (NASNetMobile, nasnet_preprocess),
            'NASNetLarge': (NASNetLarge, nasnet_preprocess),
            'VGG16': (VGG16, vgg16_preprocess),
            'VGG19': (VGG19, vgg19_preprocess)
        }
        if image_model_name not in model_map:
            raise ValueError(f"Unsupported image model: {image_model_name}")
        model_cls, preprocess_fn = model_map[image_model_name]
        model_instance = model_cls(weights='imagenet', include_top=False, pooling='avg')
        img_preprocessed = preprocess_fn(np.expand_dims(img_resized, axis=0))
        img_embed = model_instance.predict(img_preprocessed)[0]

    return {'text_embedding': text_embedding, 'image_embedding': img_embed}
