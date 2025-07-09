import numpy as np
from src.lhem import process_input
from src.tee import extract_evidence
import importlib
from plugins.plugin_loader import load_plugins

del_module = importlib.import_module('src.del')
tcen_module = importlib.import_module('src.tcen')
cdf_module = importlib.import_module('src.cdf')
sra_module = importlib.import_module('src.sra')

# Load all plugins
discovered_plugins = load_plugins('plugins')

# Dummy text
text = "OpenTrustEval is a modular evaluation framework. This result is not fabricated."

# Dummy image (random RGB)
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

print("--- LHEM: Text + Image ---")
embedding = process_input({'text': text, 'image': img}, image_model_name='EfficientNetB0')
print("Text embedding shape:", embedding['text_embedding'].shape)
print("Image embedding shape:", embedding['image_embedding'].shape)

print("--- TEE: Evidence Extraction ---")
evidence = extract_evidence(embedding)
print("Evidence vector shape:", evidence['evidence_vector'].shape)

print("--- DEL: Decision Aggregation ---")
decision = del_module.aggregate_evidence(evidence)
print("Decision score:", decision['decision_score'])
print("Decision vector shape:", decision['decision_vector'].shape)

print("--- TCEN: Explainability ---")
explanation = tcen_module.explain_decision(decision)
print("Explanation:", explanation['explanation'])

print("--- CDF: Final Decision ---")
final = cdf_module.finalize_decision(explanation)
print("Final Decision:", final['final_decision'])

print("--- SRA: Optimization ---")
optimized = sra_module.optimize_result(final)
print("Optimized Decision:", optimized['optimized_decision'])

print("--- Plugins: All Discovered Plugins ---")
for name, plugin in discovered_plugins.items():
    if hasattr(plugin, 'custom_plugin'):
        plugin_output = plugin.custom_plugin(optimized)
        print(f"Plugin [{name}] Output:", plugin_output.get('plugin_output'))
    if hasattr(plugin, 'hallucination_detector_plugin'):
        plugin_output = plugin.hallucination_detector_plugin(optimized)
        print(f"Plugin [{name}] Hallucination Output:", plugin_output.get('plugin_output'), '| Flag:', plugin_output.get('hallucination_flag'))

print("--- LHEM: Text Only ---")
embedding_text = process_input({'text': text, 'image': None})
print("Text-only embedding shape:", embedding_text['text_embedding'].shape)
print("Image embedding (should be None):", embedding_text['image_embedding'])

evidence_text = extract_evidence(embedding_text)
print("Evidence vector (text only) shape:", evidence_text['evidence_vector'].shape)

decision_text = del_module.aggregate_evidence(evidence_text)
print("Decision score (text only):", decision_text['decision_score'])
print("Decision vector (text only) shape:", decision_text['decision_vector'].shape)

explanation_text = tcen_module.explain_decision(decision_text)
print("Explanation (text only):", explanation_text['explanation'])

final_text = cdf_module.finalize_decision(explanation_text)
print("Final Decision (text only):", final_text['final_decision'])

print("--- SRA: Optimization (text only) ---")
optimized_text = sra_module.optimize_result(final_text)
print("Optimized Decision (text only):", optimized_text['optimized_decision'])

print("--- Plugins: All Discovered Plugins (text only) ---")
for name, plugin in discovered_plugins.items():
    if hasattr(plugin, 'custom_plugin'):
        plugin_output_text = plugin.custom_plugin(optimized_text)
        print(f"Plugin [{name}] Output (text only):", plugin_output_text.get('plugin_output'))
    if hasattr(plugin, 'hallucination_detector_plugin'):
        plugin_output_text = plugin.hallucination_detector_plugin(optimized_text)
        print(f"Plugin [{name}] Hallucination Output (text only):", plugin_output_text.get('plugin_output'), '| Flag:', plugin_output_text.get('hallucination_flag'))
