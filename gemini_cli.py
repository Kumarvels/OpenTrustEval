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

# Set the API key here
GEMINI_API_KEY = "AIzaSyBFk7jt2ahQjHlMaUMosyx01Lci-2-7hjg"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')


def main():
    parser = argparse.ArgumentParser(description="OpenTrustEval CLI: Evaluate trust for text and/or image input.")
    parser.add_argument('--text', type=str, help='Input text file (UTF-8)')
    parser.add_argument('--image', type=str, help='Input image file (jpg/png)')
    parser.add_argument('--image-model', type=str, default='EfficientNetB0', help='Keras image model name')
    args = parser.parse_args()

    # Read text
    if args.text:
        with open(args.text, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = None

    # Read image
    if args.image:
        img = np.array(Image.open(args.image).convert('RGB'))
    else:
        img = None

    if not text and img is None:
        print("Error: At least one of --text or --image must be provided.")
        sys.exit(1)

    # Pipeline
    input_dict = {'text': text or '', 'image': img}
    embedding = process_input(input_dict, image_model_name=args.image_model)
    evidence = extract_evidence(embedding)
    del_module = importlib.import_module('src.del')
    tcen_module = importlib.import_module('src.tcen')
    cdf_module = importlib.import_module('src.cdf')
    sra_module = importlib.import_module('src.sra')
    final = cdf_module.finalize_decision(
        tcen_module.explain_decision(
            del_module.aggregate_evidence(evidence)
        )
    )
    optimized = sra_module.optimize_result(final)

    print("\n--- Pipeline Output ---")
    print("Optimized Decision:", optimized['optimized_decision'])

    # Plugins
    discovered_plugins = load_plugins('plugins')
    for name, plugin in discovered_plugins.items():
        if hasattr(plugin, 'custom_plugin'):
            plugin_output = plugin.custom_plugin(optimized)
            print(f"Plugin [{name}] Output:", plugin_output.get('plugin_output'))


if __name__ == '__main__':
    main()
