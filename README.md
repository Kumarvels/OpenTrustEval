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
bash
git clone https://github.com/Kumarvels/OpenTrustEval.git
cd OpenTrustEval
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```



## 🌐 Launch the Unified WebUI
The WebUI provides a single interface for LLM, Data, Security, and Research management.

bash
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

## Why Trust-Based Systems Are Better Than Other Solutions (example: Label Error Detection Systems)

Let me break down the fundamental differences and explain why a trust-based approach is more comprehensive and valuable.

## Core Philosophical Differences

### **Label Error Detection System**

Focus: Data Quality → Model Performance
Approach: Find and fix problems in training data
Scope: Limited to labeled dataset issues
Outcome: Better training data


### **Trust-Based System**

Focus: Holistic System Reliability → Real-World Performance
Approach: Evaluate comprehensive trustworthiness
Scope: End-to-end system behavior including deployment
Outcome: Confidence in system behavior


## Detailed Comparison

### 1. **Scope and Coverage**

**Label Error Detection Limitations:**
python
# CleanLab approach - focused on training data
def cleanlab_approach(training_data, labels):
    # Only addresses:
    # 1. Mislabeling in training data
    # 2. Data quality issues
    # 3. Confidence in training predictions

  label_issues = find_label_errors(labels, pred_probs)
  cleaned_data = remove_label_issues(training_data, label_issues)
  return cleaned_data  # Better training data, but...
    
# What about deployment behavior? Real-world performance? 
# These are NOT addressed by label error detection alone


**Trust-Based Approach:**
python
# OpenTrustEval approach - comprehensive trust evaluation
def trust_based_approach(model, training_data, test_data, production_data):
    trust_assessment = {
        # Training Data Quality (includes label error detection)
        'data_quality': evaluate_data_quality(training_data, labels),
        
  # Model Reliability
  'reliability': evaluate_reliability(model, test_data),
        
  # Consistency Across Inputs
   'consistency': evaluate_consistency(model, various_inputs),
        
  # Fairness and Bias
   'fairness': evaluate_fairness(model, diverse_test_cases),
        
  # Robustness to Adversarial Attacks
   'robustness': evaluate_robustness(model, adversarial_examples),
        
  # Explainability and Transparency
   'explainability': evaluate_explainability(model, inputs),
        
  # Production Behavior
   'deployment_trust': evaluate_production_behavior(model, production_data)
    }
    
  return comprehensive_trust_score(trust_assessment)

 
### 2. **Real-World Performance vs. Training Performance**

**The Fundamental Problem:**
python
# Scenario: Perfect training data, poor real-world trust
class ExampleScenario:
    def demonstrate_limitation(self):
  # Training data is perfect (no label errors)
   training_data_quality = 0.99  # CleanLab would be happy
        
 # But model has issues:
   reliability_score = 0.6       # Unreliable predictions
   consistency_score = 0.5       # Inconsistent responses
   fairness_score = 0.4          # Biased decisions
   robustness_score = 0.3        # Fragile to input changes
        
 # Label error detection says: "Data is clean!"
 # Trust system says: "Don't deploy this - it's not trustworthy!"
        
   return {
            'cleanlab_assessment': 'Data quality excellent',
            'trust_assessment': 'System not ready for deployment'
        }


### 3. **Temporal and Contextual Trust**

**Label Error Detection Cannot Address:**
python
# Issues that arise over time and context
def temporal_trust_challenges():
    return {
        # Time-based issues (CleanLab can't detect):
        'concept_drift': 'Model performance degrades as world changes',
        'data_drift': 'Input distribution shifts in production',
        'model_degradation': 'Performance naturally degrades over time',
        
 # Context-based issues:
   'domain_adaptation': 'Works in training domain but fails in deployment domain',
   'edge_cases': 'Handles common cases but fails on edge cases',
   'user_trust': 'Users lose confidence due to inconsistent behavior'
    }


## Why Trust-Based Systems Are Superior

### 1. **Comprehensive Risk Assessment**

**Trust systems evaluate:**
python
def comprehensive_risk_assessment():
    return {
        # Pre-deployment risks (partially covered by CleanLab)
        'training_data_risks': ['label_errors', 'bias', 'completeness'],
        
  # Model behavior risks (NOT covered by CleanLab)
   'behavioral_risks': [
            'overconfidence',           # Model too confident in wrong answers
            'inconsistency',            # Different responses to similar inputs
            'adversarial_vulnerability', # Security risks
            'bias_amplification'        # Fairness issues in deployment
        ],
        
 # Deployment risks (NOT covered by CleanLab)
   'deployment_risks': [
            'production_drift',         # Performance degradation over time
            'user_acceptance',          # Human trust and adoption
            'regulatory_compliance',    # Legal and ethical requirements
            'business_impact'           # Real-world consequences of failures
        ]
    }


### 2. **Decision-Making Support**

**Beyond Data Quality:**
python
def decision_making_support():
    # CleanLab helps answer: "Is my training data good?"
    cleanlab_question = "Should I retrain with cleaned data?"
    
# Trust systems help answer broader questions:
  trust_questions = [
        "Should I deploy this model to production?",
        "Can I trust this model's decisions in critical situations?",
        "How will this model perform with real users?",
        "What are the risks of deploying this system?",
        "How can I improve overall system trustworthiness?"
    ]
    
 return {
        'cleanlab_scope': cleanlab_question,
        'trust_scope': trust_questions
    }


### 3. **Continuous Monitoring and Improvement**

**Evolution Over Time:**
python
def evolution_comparison():
    return {
        'label_error_detection': {
            'phase': 'Training/pre-deployment',
            'frequency': 'One-time or periodic retraining',
            'scope': 'Static training dataset',
            'outcome': 'Better training data'
        },
        
 'trust_based_system': {
            'phase': 'End-to-end lifecycle (training → deployment → monitoring)',
            'frequency': 'Continuous monitoring',
            'scope': 'Dynamic system behavior in real-world conditions',
            'outcome': 'Confidence in system reliability and safety'
        }
    }


## Concrete Examples Where Trust Systems Excel

### Example 1: **Medical Diagnosis System**

python
# CleanLab approach:
medical_model_cleanlab = {
    'training_data_quality': 0.98,  # Very clean data
    'recommendation': 'Ready for deployment'
}

# Trust-based approach:
medical_model_trust = {
    'training_data_quality': 0.98,     # Same clean data
    'reliability_score': 0.7,          # Sometimes confident when wrong
    'consistency_score': 0.6,          # Different diagnoses for similar symptoms
    'robustness_score': 0.5,           # Fragile to slight input variations
    'fairness_score': 0.8,             # Good but not perfect
    'explainability_score': 0.4,       # Poor explanations for decisions
    'overall_trust': 0.6,              # NOT ready for deployment!
    'recommendation': 'Needs significant improvement before deployment'
}


### Example 2: **Autonomous Vehicle Perception**

python
# CleanLab approach:
av_perception_cleanlab = {
    'training_data_quality': 0.95,  # Good object detection labels
    'recommendation': 'Good data quality'
}

# Trust-based approach:
av_perception_trust = {
    'training_data_quality': 0.95,     # Same good data
    'reliability_in_rain': 0.3,        # Terrible in rain conditions
    'consistency_at_night': 0.4,       # Inconsistent night performance
    'robustness_to_adversarial': 0.2,  # Vulnerable to simple attacks
    'edge_case_handling': 0.3,         # Fails on unusual scenarios
    'safety_trust': 0.3,               # DANGEROUS for deployment!
    'recommendation': 'Absolutely not ready - safety risks too high'
}


## The Trust Advantage: Beyond Binary Decisions

### **CleanLab's Binary Thinking:**

Data Quality: Good/Bad → Retrain/Don't Retrain


### **Trust-Based Thinking:**

Trust Dimensions:
├── Reliability: 0.7 (Moderate confidence)
├── Consistency: 0.6 (Some variability acceptable)
├── Fairness: 0.9 (Excellent)
├── Robustness: 0.4 (Needs improvement)
├── Explainability: 0.8 (Good)
└── Overall Trust: 0.6 (Improvement needed)

Decision Matrix:
├── Critical Applications: DON'T DEPLOY
├── Low-Stakes Applications: DEPLOY with monitoring
└── Research Applications: DEPLOY with caveats


## Fundamental Truth

**Perfect training data ≠ Trustworthy system**

A trust-based system recognizes that:
1. **Data quality is necessary but not sufficient** for trustworthy AI
2. **Model behavior in deployment matters more** than training data quality
3. **Human trust and acceptance** are crucial for real-world success
4. **Continuous monitoring and improvement** are essential for long-term success

## Conclusion

Trust-based systems are superior because they:

1. **Provide comprehensive assessment** beyond just data quality
2. **Support better decision-making** for real-world deployment
3. **Consider end-to-end system behavior** rather than isolated components
4. **Enable continuous improvement** throughout the AI lifecycle
5. **Address human factors** like user trust and acceptance
6. **Prepare for real-world complexity** rather than controlled environments

While label error detection is valuable (and should be part of any comprehensive approach), it's only one piece of the much larger trust puzzle. 
A trust-based system provides the holistic view needed to build truly reliable, safe, and successful AI systems.


## 🏆 Credits
OpenTrustEval is developed and maintained by Kumarvels and contributors. For issues, feature requests, or contributions, please open an issue or pull request on GitHub.
