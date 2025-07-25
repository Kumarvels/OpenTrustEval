# Cleanlab Datalab Plugin for OpenTrustEval

This plugin integrates [Cleanlab Datalab](https://cleanlab.ai/docs/datalab/) for automated data issue detection and review.

## Features
- Detects label errors, outliers, duplicates, and other data issues using Cleanlab Datalab.
- Optional and pluggable: only runs if Cleanlab is installed and the plugin is enabled in config.
- Designed for human-in-the-loop review and scalable data quality workflows.

## Usage
1. **Install Cleanlab (commercial license required):**
   - Add `cleanlab` to `plugins/requirements.txt` and install via pip:
     ```sh
     pip install cleanlab
     ```
   - See [Cleanlab Pricing](https://cleanlab.ai/pricing/) for licensing and consent requirements.
2. **Enable the plugin:**
   - In your config or plugin loader, set `enable_cleanlab: true` when you want to use this plugin.
3. **Call the plugin:**
   - The plugin exposes `cleanlab_datalab_plugin(data, labels=None, config=None)`.
   - Returns a dict with plugin output, detected issues, and warnings.

## Example
```python
from plugins import cleanlab_datalab_plugin
result = cleanlab_datalab_plugin([
    {'text': 'sample', 'label': 'A'},
    {'text': 'example', 'label': 'B'}
], labels=['A', 'B'], config={'enable_cleanlab': True})
print(result)
```

## Warnings & Consent
- **Commercial Use:** Cleanlab Datalab is a commercial product. You must have a valid license.
- **User Consent:** Ensure you have user consent for data analysis as required by law and Cleanlab terms.
- **Inactive by Default:** The plugin is inactive unless Cleanlab is installed and enabled in config.

## References
- [Cleanlab Datalab Documentation](https://cleanlab.ai/docs/datalab/)
- [Cleanlab Pricing](https://cleanlab.ai/pricing/) 