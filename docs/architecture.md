# OTE Architecture Overview

This document provides a detailed architecture diagram, component descriptions, and data flow for OpenTrustEval (OTE).

## Pipeline Overview

```
Text/Image Input
     │
     ▼
[LHEM] ──► [TEE] ──► [DEL] ──► [TCEN] ──► [CDF] ──► [SRA] ──► [Plugins]
     │        │         │         │         │         │         │
     ▼        ▼         ▼         ▼         ▼         ▼         ▼
Embeddings  Evidence  Decision  Explanation Final   Optimized  Custom
           Vector    Score     Output      Decision Decision   Output
```

## Module Descriptions

- **LHEM (Lightweight Hybrid Embedding Module):**
  - Extracts text and image embeddings using DistilBERT and Keras image models.
- **TEE (Trust Evidence Extraction):**
  - Combines embeddings into a unified evidence vector.
- **DEL (Decision Evidence Layer):**
  - Aggregates evidence into a decision score/vector.
- **TCEN (Explainability Engine):**
  - Generates a human-readable explanation for the decision.
- **CDF (Decision Module):**
  - Packages the explanation as a final decision.
- **SRA (Optimization Layer):**
  - Optimizes/post-processes the final decision for deployment.
- **Plugins:**
  - Dynamically loaded from the `plugins/` directory. Each plugin can further process the optimized decision.

## Plugin System

- Plugins are Python modules in the `plugins/` directory with a `custom_plugin` function.
- All plugins are auto-discovered and applied to the pipeline output.

## CLI Usage

Run the full pipeline on real data:

```sh
python3 gemini_cli.py --text sample.txt --image sample.jpg
```
- `--text`: Path to a UTF-8 text file
- `--image`: Path to an image file (jpg/png)
- `--image-model`: (Optional) Keras image model name (default: EfficientNetB0)

## Example Output

```
--- Pipeline Output ---
Optimized Decision: Finalized: The decision score -0.0302 indicates a negative trust assessment. [Optimized for deployment]
Plugin [example_plugin] Output: Finalized: The decision score -0.0302 indicates a negative trust assessment. [Optimized for deployment] [Plugin: Example note added]
```

---

For more details, see the code in each module and the `plugins/` directory.
