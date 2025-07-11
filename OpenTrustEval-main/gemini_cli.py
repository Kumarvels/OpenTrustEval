# gemini_cli.py
import argparse
import google.generativeai as genai
import numpy as np
from PIL import Image
import sys
from src.lhem import process_input
from src.tee import extract_evidence
import importlib
from plugins.plugin_loader import load_plugins
import os
import csv

# Set the API key here
GEMINI_API_KEY = "AIzaSyBFk7jt2ahQjHlMaUMosyx01Lci-2-7hjg"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')


def process_pipeline(text, img, image_model, discovered_plugins):
    input_dict = {'text': text or '', 'image': img}
    embedding = process_input(input_dict, image_model_name=image_model)
    del_module = importlib.import_module('src.del')
    tcen_module = importlib.import_module('src.tcen')
    cdf_module = importlib.import_module('src.cdf')
    sra_module = importlib.import_module('src.sra')
    evidence = extract_evidence(embedding)
    final = cdf_module.finalize_decision(
        tcen_module.explain_decision(
            del_module.aggregate_evidence(evidence)
        )
    )
    optimized = sra_module.optimize_result(final)
    plugin_outputs = {}
    for name, plugin in discovered_plugins.items():
        if hasattr(plugin, 'custom_plugin'):
            plugin_outputs[f'{name}_custom'] = plugin.custom_plugin(optimized)
        if hasattr(plugin, 'hallucination_detector_plugin'):
            plugin_outputs[f'{name}_hallucination'] = plugin.hallucination_detector_plugin(optimized)
    return optimized, plugin_outputs


def read_text_file(path):
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm']:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    elif ext in ['.npy']:
        return str(np.load(path))
    else:
        raise ValueError(f"Unsupported text file format: {ext}")


def read_image_file(path):
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
        return np.array(Image.open(path).convert('RGB'))
    elif ext in ['.npy']:
        arr = np.load(path)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        return arr
    else:
        raise ValueError(f"Unsupported image file format: {ext}")


def main():
    parser = argparse.ArgumentParser(description="OpenTrustEval CLI: Evaluate trust for text and/or image input, or batch mode.")
    parser.add_argument('--text', type=str, help='Input text file (UTF-8)')
    parser.add_argument('--image', type=str, help='Input image file (jpg/png)')
    parser.add_argument('--image-model', type=str, default='EfficientNetB0', help='Keras image model name')
    parser.add_argument('--batch-csv', type=str, help='CSV file with columns: text_path,image_path')
    args = parser.parse_args()

    discovered_plugins = load_plugins('plugins')

    if args.batch_csv:
        with open(args.batch_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                text = read_text_file(row['text_path']) if row['text_path'] else None
                img = read_image_file(row['image_path']) if row['image_path'] else None
                print(f"\n--- Batch Item {i+1} ---")
                optimized, plugin_outputs = process_pipeline(text, img, args.image_model, discovered_plugins)
                print("Optimized Decision:", optimized['optimized_decision'])
                for k, v in plugin_outputs.items():
                    print(f"Plugin [{k}] Output:", v)
        return

    # Single input mode
    text = read_text_file(args.text) if args.text else None
    img = read_image_file(args.image) if args.image else None
    if not text and img is None:
        print("Error: At least one of --text or --image must be provided.")
        sys.exit(1)
    optimized, plugin_outputs = process_pipeline(text, img, args.image_model, discovered_plugins)
    print("\n--- Pipeline Output ---")
    print("Optimized Decision:", optimized['optimized_decision'])
    for k, v in plugin_outputs.items():
        print(f"Plugin [{k}] Output:", v)


if __name__ == '__main__':
    main()
