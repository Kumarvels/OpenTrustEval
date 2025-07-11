import os

REQUIRED_FILES = [
    'config.json',
    'generation_config.json',
    'model-00001-of-00002.safetensors',
    'model-00002-of-00002.safetensors',
    'model.safetensors.index.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.json',
    'merges.txt',
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'phi-4-mini-flash-reasoning')

missing = []
zero_length = []

for fname in REQUIRED_FILES:
    fpath = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(fpath):
        missing.append(fname)
    elif os.path.getsize(fpath) == 0:
        zero_length.append(fname)

if not missing and not zero_length:
    print("All required Phi-4-mini-reasoning model files are present and non-empty.")
else:
    if missing:
        print("Missing files:")
        for f in missing:
            print("  -", f)
    if zero_length:
        print("Zero-length files (corrupt or incomplete):")
        for f in zero_length:
            print("  -", f) 