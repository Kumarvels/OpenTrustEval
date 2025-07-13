from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi import Body
import numpy as np
from PIL import Image
import io
from high_performance_system.legacy_compatibility import process_input
from high_performance_system.legacy_compatibility import extract_evidence
import importlib
from plugins.plugin_loader import load_plugins
import time
from typing import List, Optional
import asyncio
import psutil
from high_performance_system.legacy_compatibility import init_db, log_pipeline

# Import LLM manager router
try:
    from cloudscale_apis.endpoints.llm_manager import router as llm_router
    LLM_ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LLM router not available: {e}")
    LLM_ROUTER_AVAILABLE = False

# Import Data manager router
try:
    from cloudscale_apis.endpoints.data_manager import router as data_router
    DATA_ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data router not available: {e}")
    DATA_ROUTER_AVAILABLE = False

# Import Security manager router
try:
    from cloudscale_apis.endpoints.security_manager import router as security_router
    SECURITY_ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Security router not available: {e}")
    SECURITY_ROUTER_AVAILABLE = False

# Import Research Platform router
try:
    from cloudscale_apis.endpoints.research_platform import router as research_router
    RESEARCH_ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Research router not available: {e}")
    RESEARCH_ROUTER_AVAILABLE = False

app = FastAPI(title="OpenTrustEval API")

# Include LLM router if available
if LLM_ROUTER_AVAILABLE:
    app.include_router(llm_router)
    print("✅ LLM Management API endpoints included")

# Include Data router if available
if DATA_ROUTER_AVAILABLE:
    app.include_router(data_router)
    print("✅ Data Management API endpoints included")

# Include Security router if available
if SECURITY_ROUTER_AVAILABLE:
    app.include_router(security_router)
    print("✅ Security Management API endpoints included")

# Include Research router if available
if RESEARCH_ROUTER_AVAILABLE:
    app.include_router(research_router)
    print("✅ Research Platform API endpoints included")

# Model cache
distilbert_tokenizer = None
distilbert_model = None
keras_models = {}

def process_input_cached(input_data, image_model_name='EfficientNetB0'):
    global distilbert_tokenizer, distilbert_model, keras_models
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    import numpy as np
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

    # Text embedding (cache DistilBERT)
    if distilbert_tokenizer is None or distilbert_model is None:
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        distilbert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    tokenizer = distilbert_tokenizer
    model = distilbert_model
    inputs = tokenizer(input_data['text'], return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Image embedding (cache Keras models)
    img_embed = None
    if input_data.get('image') is not None:
        img = input_data['image']
        if img.shape[-1] != 3:
            raise ValueError('Image must have 3 channels (RGB)')
        img_resized = keras_image.smart_resize(img, (224, 224))
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
        if image_model_name not in keras_models:
            model_cls, _ = model_map[image_model_name]
            keras_models[image_model_name] = model_cls(weights='imagenet', include_top=False, pooling='avg')
        model_instance = keras_models[image_model_name]
        _, preprocess_fn = model_map[image_model_name]
        img_preprocessed = preprocess_fn(np.expand_dims(img_resized, axis=0))
        img_embed = model_instance.predict(img_preprocessed)[0]
    return {'text_embedding': text_embedding, 'image_embedding': img_embed}

def process_pipeline(text, img, image_model, discovered_plugins):
    timings = {}
    t0 = time.time()
    input_dict = {'text': text or '', 'image': img}
    embedding = process_input_cached(input_dict, image_model_name=image_model)
    timings['lhem'] = time.time() - t0
    t1 = time.time()
    del_module = importlib.import_module('src.del')
    tcen_module = importlib.import_module('src.tcen')
    cdf_module = importlib.import_module('src.cdf')
    sra_module = importlib.import_module('src.sra')
    evidence = extract_evidence(embedding)
    timings['tee'] = time.time() - t1
    t2 = time.time()
    decision = del_module.aggregate_evidence(evidence)
    timings['del'] = time.time() - t2
    t3 = time.time()
    explanation = tcen_module.explain_decision(decision)
    timings['tcen'] = time.time() - t3
    t4 = time.time()
    final = cdf_module.finalize_decision(explanation)
    timings['cdf'] = time.time() - t4
    t5 = time.time()
    optimized = sra_module.optimize_result(final)
    timings['sra'] = time.time() - t5
    t6 = time.time()
    plugin_outputs = {}
    for name, plugin in discovered_plugins.items():
        if hasattr(plugin, 'custom_plugin'):
            plugin_outputs[f'{name}_custom'] = plugin.custom_plugin(optimized)
        if hasattr(plugin, 'hallucination_detector_plugin'):
            plugin_outputs[f'{name}_hallucination'] = plugin.hallucination_detector_plugin(optimized)
    timings['plugins'] = time.time() - t6
    timings['total'] = time.time() - t0
    return optimized, plugin_outputs, timings

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/evaluate/")
async def evaluate(
    text: str = Form(None),
    image: UploadFile = File(None),
    image_model: str = Form('EfficientNetB0')
):
    discovered_plugins = load_plugins('plugins')
    img = None
    try:
        if image is not None:
            img_bytes = await image.read()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
        optimized, plugin_outputs, timings = process_pipeline(text, img, image_model, discovered_plugins)
        process = psutil.Process()
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        resource = {
            'memory_rss_mb': mem_info.rss // (1024 * 1024),
            'cpu_percent': cpu_percent
        }
        log_pipeline('pipeline_logs.db',
            input_type='text+image' if text and img is not None else 'text' if text else 'image',
            timings=timings,
            resource=resource,
            optimized_decision=optimized['optimized_decision'],
            plugin_outputs=plugin_outputs,
            error=None
        )
        return JSONResponse({
            'optimized_decision': optimized['optimized_decision'],
            'plugins': plugin_outputs,
            'timings': timings,
            'resource_usage': resource
        })
    except Exception as e:
        log_pipeline('pipeline_logs.db',
            input_type='text+image' if text and img is not None else 'text' if text else 'image',
            timings={},
            resource={'memory_rss_mb': 0, 'cpu_percent': 0},
            optimized_decision='',
            plugin_outputs={},
            error=str(e)
        )
        return JSONResponse({'error': str(e)}, status_code=400)

@app.post("/batch_evaluate/")
async def batch_evaluate(
    items: List[dict] = Body(..., example=[{"text": "sample text", "image": None, "image_model": "EfficientNetB0"}])
):
    discovered_plugins = load_plugins('plugins')
    async def process_item(item):
        try:
            text = item.get('text')
            img = None
            if item.get('image'):
                img_bytes = io.BytesIO(bytes(item['image']))
                img = np.array(Image.open(img_bytes).convert('RGB'))
            image_model = item.get('image_model', 'EfficientNetB0')
            optimized, plugin_outputs, timings = process_pipeline(text, img, image_model, discovered_plugins)
            process = psutil.Process()
            mem_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            resource = {
                'memory_rss_mb': mem_info.rss // (1024 * 1024),
                'cpu_percent': cpu_percent
            }
            log_pipeline('pipeline_logs.db',
                input_type='text+image' if text and img is not None else 'text' if text else 'image',
                timings=timings,
                resource=resource,
                optimized_decision=optimized['optimized_decision'],
                plugin_outputs=plugin_outputs,
                error=None
            )
            return {
                'optimized_decision': optimized['optimized_decision'],
                'plugins': plugin_outputs,
                'timings': timings,
                'resource_usage': resource
            }
        except Exception as e:
            log_pipeline('pipeline_logs.db',
                input_type='text+image' if item.get('text') and item.get('image') is not None else 'text' if item.get('text') else 'image',
                timings={},
                resource={'memory_rss_mb': 0, 'cpu_percent': 0},
                optimized_decision='',
                plugin_outputs={},
                error=str(e)
            )
            return {'error': str(e)}
    results = await asyncio.gather(*(process_item(item) for item in items))
    return JSONResponse({'results': results})

@app.post("/realtime_evaluate/")
async def realtime_evaluate(
    text: str = Form(...),
    image: UploadFile = File(None),
    image_model: str = Form('EfficientNetB0'),
    min_trust: str = Form('Trustworthy')
):
    """
    Fast endpoint for chatbot/LLM integration. Returns minimal trust/hallucination result for real-time gating.
    """
    discovered_plugins = load_plugins('plugins')
    img = None
    try:
        if image is not None:
            img_bytes = await image.read()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
        optimized, plugin_outputs, timings = process_pipeline(text, img, image_model, discovered_plugins)
        # Minimal response for real-time use
        halluc_flag = False
        for v in plugin_outputs.values():
            if isinstance(v, dict) and v.get('hallucination_flag'):
                halluc_flag = True
        trust_score = optimized['optimized_decision']
        allow = (trust_score == min_trust) and not halluc_flag
        return {
            'allow': allow,
            'trust_score': trust_score,
            'hallucination': halluc_flag,
            'plugin_outputs': plugin_outputs
        }
    except Exception as e:
        return {'error': str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}
