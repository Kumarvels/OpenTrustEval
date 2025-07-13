# OpenTrustEval: Unified Trustworthy AI Evaluation Platform

## 🚀 Overview
OpenTrustEval is a comprehensive, high-performance, and modular platform for AI evaluation, hallucination detection, data quality, and trustworthy AI lifecycle management. It integrates:
- **LLM Management** (dynamic loading, fine-tuning, evaluation, deployment)
- **Data Engineering** (ETL, trust scoring, validation, analytics)
- **Security** (auth, monitoring, secrets, OAuth/SAML)
- **Research Platform** (experiments, use cases, analysis, project management)
- **Unified WebUI** for all managers and workflows
- **Super-fast async API** for all components

---

## 📦 Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git
- [Optional] Node.js (for advanced dashboard features)

---

## 🛠️ Installation
```bash
git clone https://github.com/Kumarvels/OpenTrustEval.git
cd OpenTrustEval
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

---

## 🌐 Launch the Unified WebUI
The WebUI provides a single interface for LLM, Data, Security, and Research management.

```bash
streamlit run launch_workflow_webui.py
```
- Open [http://localhost:8501](http://localhost:8501) in your browser.
- All managers and dashboards are available from the sidebar.

---

## 🚦 Launch the Production API Server
The API server exposes all async endpoints for programmatic access.

```bash
python superfast_production_server.py
```
- The server runs at [http://localhost:8003](http://localhost:8003)
- Health check: [http://localhost:8003/health](http://localhost:8003/health)

---

## 🧭 Step-by-Step Platform Flow
1. **Start the API server** (`python superfast_production_server.py`)
2. **Start the WebUI** (`streamlit run launch_workflow_webui.py`)
3. **Upload or create datasets** (Data Manager)
4. **Run ETL, trust scoring, and validation** (Data Manager)
5. **Manage and deploy LLMs** (LLM Manager)
6. **Configure security, users, and secrets** (Security Manager)
7. **Create research use cases and experiments** (Research Platform)
8. **Analyze results, generate reports, and monitor system** (WebUI dashboards)

---

## 🧩 Manager Integration
- **LLM Manager**: CRUD, fine-tune, evaluate, deploy, batch ops (`/llm/*`)
- **Data Manager**: Dataset CRUD, trust scoring, ETL, batch ops (`/data/*`)
- **Security Manager**: Auth, user mgmt, monitoring, secrets, OAuth/SAML (`/security/*`)
- **Research Platform**: Use cases, experiments, analysis, project mgmt (`/research/*`)
- **Unified WebUI**: All managers, dashboards, and workflows in one place

---

## 🛠️ Example Usage Flows
### **A. WebUI**
- Launch: `streamlit run launch_workflow_webui.py`
- Use sidebar to:
  - Upload data, run trust scoring, view analytics
  - Manage LLMs (load, fine-tune, deploy)
  - Configure security, users, secrets
  - Create and run research experiments

### **B. API**
- Health: `GET /health`
- List datasets: `GET /data/datasets/list`
- Create dataset: `POST /data/datasets/create`
- Trust score: `POST /data/datasets/{dataset_id}/trust-score`
- LLM health: `GET /llm/health`
- Security health: `GET /security/health`
- Research health: `GET /research/health`

---

## 🧑‍💻 Troubleshooting & Tips
- **Missing dependencies?** Run `pip install -r requirements.txt` again.
- **Port in use?** Change the port in the server script or stop the conflicting process.
- **Module not found?** Ensure your virtual environment is activated.
- **Windows line endings warning?** Safe to ignore, or run `git config --global core.autocrlf true`.
- **API 404?** Make sure the server is running and the endpoint path is correct.
- **WebUI not loading?** Check for errors in the terminal and ensure Streamlit is installed.

---

## 📚 More Documentation
- See `DEEP_INTEGRATION_ANALYSIS.md` for cross-component workflows
- See `COMPLETE_WORKFLOW_SUMMARY.md` for end-to-end diagnostic flows
- See `README_UNIFIED_WEBUI.md` for WebUI usage

---

## 🏆 Credits
OpenTrustEval is developed and maintained by Kumarvels and contributors. For issues, feature requests, or contributions, please open an issue or pull request on GitHub.
