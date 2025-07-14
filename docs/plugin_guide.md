# Plugin Development Guide

Step-by-step guide for plugin creation, API specs, and submission process.

## Cleanlab Datalab Plugin (Optional, Commercial/Consent Required)

- **Purpose:** Detects data issues (label errors, outliers, duplicates) using Cleanlab Datalab.
- **Location:** `plugins/cleanlab_datalab_plugin.py`
- **Activation:** Only runs if Cleanlab is installed and `enable_cleanlab: true` is set in config.
- **Commercial/Consent:** Requires a Cleanlab Datalab license and user consent. See [Cleanlab Pricing](https://cleanlab.ai/pricing/).
- **Usage:**
  - Install Cleanlab as needed: `pip install cleanlab` (see `plugins/requirements.txt`)
  - Enable in config or plugin loader when required.
  - See `plugins/README_CLEANLAB_PLUGIN.md` for details and code examples.
- **Integration:** Used in data engineering pipelines (see `project_building_data_model_etl/project_etl.py`). Enable via CLI (`--enable-cleanlab`) or config. See pipeline README for details.
