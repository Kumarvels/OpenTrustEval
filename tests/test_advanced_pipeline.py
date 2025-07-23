import numpy as np
import pytest
from high_performance_system.legacy_compatibility import process_input
from high_performance_system.legacy_compatibility import extract_evidence
import importlib

del_module = importlib.import_module('src.del')
tcen_module = importlib.import_module('src.tcen')
cdf_module = importlib.import_module('src.cdf')
sra_module = importlib.import_module('src.sra')
from plugins.plugin_loader import load_plugins

@pytest.mark.parametrize("text,img_shape", [
    ("", (256, 256, 3)),  # Empty text, valid image
    ("Test text only", None),  # Text only
    ("", None),  # Neither text nor image (should error)
    ("Test", (256, 256, 1)),  # Invalid image shape
])
def test_pipeline_edge_cases(text, img_shape):
    img = None
    if img_shape:
        img = np.random.randint(0, 255, img_shape, dtype=np.uint8)
    input_dict = {'text': text, 'image': img}
    if not text and img is None:
        try:
            process_input(input_dict)
        except Exception as e:
            assert isinstance(e, Exception)
        else:
            assert False, "Should raise error for no input"
        return
    if img is not None and (img.shape[-1] != 3):
        try:
            process_input(input_dict)
        except ValueError as e:
            assert '3 channels' in str(e)
        else:
            assert False, "Should raise error for invalid image shape"
        return
    embedding = process_input(input_dict)
    evidence = extract_evidence(embedding)
    decision = del_module.aggregate_evidence(evidence)
    explanation = tcen_module.explain_decision(decision)
    final = cdf_module.finalize_decision(explanation)
    optimized = sra_module.optimize_result(final)
    discovered_plugins = load_plugins('plugins')
    for name, plugin in discovered_plugins.items():
        if hasattr(plugin, 'custom_plugin'):
            plugin_output = plugin.custom_plugin(optimized)
            assert 'plugin_output' in plugin_output
