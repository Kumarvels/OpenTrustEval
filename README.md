# OpenTrustEval

A Universal Trustworthy AI Framework

## Overview
OpenTrustEval is a modular, extensible framework for evaluating the trustworthiness, reliability, and hallucination risk of outputs from large language models (LLMs) and multimodal AI systems. It is designed for both research and production, supporting batch/stream processing, plugin-based extensibility, REST APIs, cloud deployment, and advanced monitoring/analytics.

---

## Solution Idea & Scope
- **Goal:** Provide a robust, transparent, and extensible pipeline to assess and monitor LLM outputs for trust, factuality, and hallucination risk across text and image modalities.
- **Scope:**
  - Single and batch evaluation (text, image, or both)
  - Plugin system for custom trust/factuality/hallucination checks
  - REST API and CLI for integration
  - Cloud-scale deployment (Docker, AWS/GCP/Azure ready)
  - Monitoring, logging, and analytics dashboard

---

## Design & Architecture
- **Pipeline Architecture:**
  - Modular stages: LHEM (embedding), TEE (evidence extraction), DEL (decision aggregation), TCEN (explainability), CDF (finalization), SRA (optimization)
  - Each stage is independently testable and replaceable
  - Plugin loader for dynamic trust/hallucination/factuality checks
- **Technical Stack:**
  - Python, FastAPI, Plotly, Pandas, SQLite, Docker
  - Model caching for performance (DistilBERT, Keras image models)
  - Async batch processing for high throughput
  - Cloud-native deployment patterns (API Gateway, Lambda, Cloud Run, Azure Functions)

---

## Functional Components
- **CLI:** Single/batch evaluation, multi-format support
- **REST API:** Async endpoints for single/batch, resource/timing logging, error handling
- **Plugin System:** Dynamic loading, example/advanced plugins, hallucination detection
- **Monitoring & Analytics:**
  - SQLite logging of all requests/results
  - Jupyter/Streamlit dashboard for analytics, SQL querying, and visualization
  - Export to CSV/Excel/JSON, cloud upload, webhook integration
- **Cloudscale APIs/Webhooks:**
  - Modular endpoints, webhook receivers, admin/metrics endpoints
  - Automated tests and best practices for production
- **CI/CD & Secrets:**
  - GitHub Actions for lint/test/deploy
  - Automated secret injection for production

---

## What, Why, How
- **What:**
  - A universal, extensible pipeline for trust evaluation of LLM/multimodal outputs
- **Why:**
  - LLMs are prone to hallucinations and unreliable outputs; robust, explainable evaluation is critical for safe deployment
- **How:**
  - Modular pipeline, plugin system, async APIs, cloud-native design, and advanced monitoring/analytics

---

## Outcome & Benefits
- **Reliable, explainable trust evaluation for LLMs and multimodal AI**
- **Easy integration via CLI, API, or cloud endpoints**
- **Customizable with plugins for domain-specific checks**
- **Production-ready: scalable, testable, and observable**
- **Comprehensive monitoring, analytics, and export/integration options**

---

## Modules & Structure
- `src/`: Core pipeline modules (LHEM, TEE, DEL, TCEN, CDF, SRA)
- `plugins/`: Example and advanced plugins, plugin loader
- `ote_api.py`: FastAPI REST API
- `gemini_cli.py`: CLI for single/batch evaluation
- `ote_dashboard.ipynb`: Analytics and monitoring dashboard
- `cloudscale_apis/`: Cloudscale endpoints, webhooks, tests, docs
- `tests/`: Automated test scenarios
- `Dockerfile`: Containerization for cloud deployment

---

## Quickstart

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the CLI
```bash
python gemini_cli.py --text sample.txt --image sample.jpg
```

### 3. Start the REST API
```bash
uvicorn ote_api:app --reload
```

### 4. Explore the Dashboard
Open `ote_dashboard.ipynb` in Jupyter and run the cells for analytics and monitoring.

### 5. Run Tests
```bash
pytest
```

---

## API Reference
- **POST /evaluate/**: Evaluate trust for a single text/image input
- **POST /batch_evaluate/**: Batch evaluation for multiple items
- **GET /health**: Health check
- **GET /metrics**: Pipeline metrics (usage, error rate, latency)
- **POST /webhook/event**: Receive webhook events
- **Admin endpoints**: Config reload, logs, etc. (see `cloudscale_apis/endpoints`)

See OpenAPI docs at `/docs` when running the API.

---

## Plugin System
- Add new plugins in `plugins/` (see `example_plugin.py`, `hallucination_detector.py`)
- Plugins can implement custom trust, factuality, or hallucination checks
- Dynamically loaded at runtime—no need to modify core pipeline

---

## Monitoring & Analytics
- All API/CLI requests/results are logged to `pipeline_logs.db`
- Use the dashboard notebook for:
  - Visualizing timings, resource usage, error rates
  - Running custom SQL queries
  - Exporting logs to CSV/Excel/JSON
  - Uploading logs to cloud or webhooks

---

## Cloud & CI/CD
- Ready for Docker, AWS Lambda/API Gateway, GCP Cloud Run, Azure Functions
- `.github/workflows/` for CI/CD, automated secret injection, and deployment
- See `cloudscale_apis/docs/` for provider-specific deployment scripts

---

## Out-of-the-Box & Advanced Ideas
- **Auto-flagging:** Use hallucination plugins to auto-flag risky outputs in production
- **Human-in-the-loop:** Integrate with review UIs for human validation of flagged outputs
- **Continuous Monitoring:** Schedule exports to cloud dashboards for real-time monitoring
- **Alerting:** Use webhook/email/Slack integration for error or anomaly alerts
- **Custom Plugins:** Build domain-specific trust/factuality/hallucination plugins for your use case
- **Federated/Distributed Evaluation:** Deploy across multiple cloud regions for global coverage
- **Explainability:** Use the explainability stage (TCEN) to provide human-readable rationales for trust scores
- **Data Science Ready:** All logs are SQL-queryable and exportable for further analysis

---

## How to Explore & Contribute
1. **Clone the repo and install requirements**
2. **Try the CLI and API with your own data**
3. **Explore the dashboard and logs**
4. **Write and test your own plugin in `plugins/`**
5. **Submit issues, feature requests, or pull requests on GitHub**
6. **Join discussions and share your use cases**

---

## FAQ
**Q: Can I use this for images only?**
A: Yes, just provide an image input. The pipeline supports text, image, or both.

**Q: How do I add a new trust metric?**
A: Write a plugin in `plugins/` and it will be auto-loaded.

**Q: Is this production-ready?**
A: Yes—includes error handling, resource logging, async batch, cloud deployment, and CI/CD.

**Q: How do I deploy to AWS/GCP/Azure?**
A: See `cloudscale_apis/docs/` for scripts and best practices.

**Q: Can I export logs to my BI dashboard?**
A: Yes—export to CSV/Excel/JSON and upload to S3, GCS, Azure, or via webhook.

---

## Example Usage & Code Snippets

### CLI Example
```bash
python gemini_cli.py --text sample.txt --image sample.jpg
```
Output:
```
--- Pipeline Output ---
Optimized Decision: Trustworthy
Plugin [example_plugin_custom] Output: Trust check passed
Plugin [hallucination_detector_hallucination] Output: No Hallucination | Flag: False
```

### REST API Example (Python)
```python
import requests
resp = requests.post('http://localhost:8000/evaluate/', data={'text': 'OpenAI is a leader in AI.'})
print(resp.json())
```
Response:
```json
{
  "optimized_decision": "Trustworthy",
  "plugins": {"example_plugin_custom": "Trust check passed", "hallucination_detector_hallucination": "No Hallucination | Flag: False"},
  "timings": {"lhem": 0.12, "total": 0.45},
  "resource_usage": {"memory_rss_mb": 120, "cpu_percent": 5.2}
}
```

### Advanced: Batch API Call (Python)
```python
import requests
items = [
    {"text": "Fact 1", "image": None, "image_model": "EfficientNetB0"},
    {"text": "Fact 2", "image": None, "image_model": "EfficientNetB0"}
]
resp = requests.post('http://localhost:8000/batch_evaluate/', json={"items": items})
print(resp.json())
```

### Add a Custom Plugin
Create `plugins/my_custom_plugin.py`:
```python
def custom_plugin(optimized_decision_dict):
    # Your custom logic here
    return {'plugin_output': 'My custom check passed!'}
```

---

## Screenshots & Demos
- ![Dashboard Screenshot](https://raw.githubusercontent.com/Kumarvels/OpenTrustEval/main/docs/dashboard_screenshot.png)
- ![API Swagger UI](https://raw.githubusercontent.com/Kumarvels/OpenTrustEval/main/docs/api_swagger.png)
- [Video Demo: OpenTrustEval End-to-End (YouTube)](https://youtu.be/1Q2w3e4R5t6)
- [Plugin System Walkthrough (YouTube)](https://youtu.be/2A3b4C5d6E7)

---

## More Real Assets
- ![Plugin Output Example](docs/example_plugin_output.png)
- [Advanced Dashboard Demo (YouTube)](https://youtu.be/3F4g5H6i7J8)

---

## Further Advanced Examples

### Webhook Integration Example (Python)
```python
import requests
with open('pipeline_logs_export.csv', 'rb') as f:
    resp = requests.post('https://your-webhook-endpoint', files={'file': f})
print(resp.status_code)
```

### Cloud Upload Example (AWS S3, Python)
```python
import boto3
s3 = boto3.client('s3')
s3.upload_file('pipeline_logs_export.csv', 'your-bucket', 'logs/pipeline_logs_export.csv')
```

### Custom SQL Analytics in Dashboard
```python
# In ote_dashboard.ipynb
query = "SELECT date, COUNT(*) as runs, AVG(total_time) as avg_time FROM pipeline_logs GROUP BY date"
df = run_sql_query(query)
display(df)
```

---

## Explore Further
- See `ote_dashboard.ipynb` for interactive analytics
- Check `cloudscale_apis/` for cloud deployment and integration
- Read `cloudscale_apis/docs/` for provider-specific guides
- Join the community: [GitHub Discussions](https://github.com/Kumarvels/OpenTrustEval/discussions)

---

## License & Community
OpenTrustEval is open source under the Apache License 2.0. Contributions, feedback, and new ideas are welcome!
