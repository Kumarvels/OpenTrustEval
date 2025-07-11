import numpy as np
from PIL import Image
from src.lhem import process_input

# Dummy text
text = "OpenTrustEval is a modular evaluation framework."

# Dummy image (random RGB)
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

# Test with both text and image
result = process_input({'text': text, 'image': img}, image_model_name='EfficientNetB0')
print("Text embedding shape:", result['text_embedding'].shape)
print("Image embedding shape:", result['image_embedding'].shape)

# Test with text only
result_text = process_input({'text': text, 'image': None})
print("Text-only embedding shape:", result_text['text_embedding'].shape)
print("Image embedding (should be None):", result_text['image_embedding'])