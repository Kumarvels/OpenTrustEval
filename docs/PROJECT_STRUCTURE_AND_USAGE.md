# OpenTrustEval Project Structure & Purpose

## Top-Level Folders

- **data_engineering/**
  - **Purpose:** All data/ETL pipeline logic, connectors, and project examples.
  - **How/Why:** Centralizes all data engineering logic, making it easy to add new projects, connectors, or pipelines. Each subfolder is a self-contained project.

- **plugins/**
  - **Purpose:** Pluggable modules for GDPR, hallucination detection, dialect, etc.
  - **How/Why:** Enables easy extension of pipeline features without modifying core code. Drop-in new plugins for new features.

- **llm_engineering/**
  - **Purpose:** LLM lifecycle, pluggable LLM providers.
  - **How/Why:** Modularizes LLM-related logic, making it easy to swap or extend LLM providers.

- **cloudscale_apis/**
  - **Purpose:** Cloud API docs, endpoints, and tests.
  - **How/Why:** Keeps cloud integration logic and documentation separate for clarity and maintainability.

- **thirdparty_integrations/**
  - **Purpose:** Adapters, endpoints, and webhooks for external systems.
  - **How/Why:** Clean separation of integration logic for third-party tools and platforms.

- **src/**
  - **Purpose:** Core algorithms, batch input, and pipeline utilities.
  - **How/Why:** Houses reusable core logic and utilities used across the project.

- **tests/**
  - **Purpose:** All test scripts.
  - **How/Why:** Centralizes all testing logic for easy CI/CD and quality assurance.

- **docs/**
  - **Purpose:** Documentation, architecture, changelogs, screenshots.
  - **How/Why:** All documentation in one place for onboarding and reference.

## Example: data_engineering/project_covid_data_analysis/

- **data/**
  - **Purpose:** Stores raw or synthetic Covid-19 data.
  - **How/Why:** Keeps data files organized and separate from code.
- **generate_synthetic_covid_data.py**
  - **Purpose:** Script to generate synthetic Covid-19 data for testing.
  - **How/Why:** Enables robust testing and demoing of the pipeline without real data.
- **project_covid_etl.py**
  - **Purpose:** Main ETL pipeline script for Covid-19 data.
  - **How/Why:** Orchestrates the full data pipeline, using shared loaders and connectors.
- **README.md**
  - **Purpose:** Project-specific documentation and instructions.
  - **How/Why:** Helps users understand and run the project quickly.

## How This Structure Helps
- **Modularity:** Each project, connector, and plugin is self-contained and easy to extend.
- **Reusability:** Shared utilities (like the data loader) prevent code duplication.
- **Clarity:** Clear folder and file naming, with documentation, makes onboarding easy.
- **Extensibility:** New projects, connectors, or plugins can be added with minimal changes to existing code.

---

# How to Add a New Project
1. Create a new folder under `data_engineering/`.
2. Add a `data/` subfolder, ETL script, and README.
3. Use `common_data_loader.py` for all data loading.

# How to Add a New Connector or Plugin
- Add a new file in `connectors/` or `plugins/` and register it in the loader.

# Example: Using the Shared Loader
```python
from data_engineering.common_data_loader import resolve_data_path, robust_read_csv
parser = argparse.ArgumentParser()
parser.add_argument('--local-path', type=str)
args, _ = parser.parse_known_args()
data_path = resolve_data_path('data/my_data.csv', args)
df = robust_read_csv(data_path)
```

# See also
- [project_structure.dot](project_structure.dot) for a Graphviz diagram of the project structure.
