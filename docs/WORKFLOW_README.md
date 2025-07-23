# 🔍 Complete Workflow Diagnostic & Problem Resolution System

## 🎯 Overview

This comprehensive workflow system provides step-by-step problem analysis and resolution for the entire OpenTrustEval platform. It covers all components from data uploads to production deployment, ensuring your system runs smoothly and efficiently.

## 📋 System Components Covered

### 🔧 Core Components
- **System Environment** - Python, dependencies, system resources
- **Data Uploads** - File handling, permissions, dataset management
- **Data Engineering** - ETL pipelines, trust scoring, database connectivity
- **LLM Engineering** - Model management, providers, lifecycle
- **High Performance System** - MoE system, expert ensemble, performance
- **Security System** - Authentication, authorization, security features

### 🌐 Network & Services
- **MCP Server** - Model Context Protocol server
- **Production Server** - Main API server
- **Cloud APIs** - Cloud provider integrations
- **Third-party Integrations** - External API connections

### 🧪 Quality Assurance
- **Tests** - System test suites
- **Plugins** - Plugin system functionality
- **Analytics & Dashboards** - Monitoring and visualization

## 🚀 Quick Start

### 1. Run Complete Diagnostic
```bash
# Run comprehensive system check
python workflow_launcher.py --diagnostic

# Or use the interactive launcher
python workflow_launcher.py
```

### 2. Resolve Problems
```bash
# Interactive problem resolution
python workflow_launcher.py --resolve

# Or run the resolver directly
python workflow_problem_resolver.py
```

### 3. Start Services
```bash
# Start production server
python workflow_launcher.py --server

# Launch dashboards
python workflow_launcher.py
# Then select option 4
```

## 📊 Diagnostic Workflow

### Step-by-Step Analysis

1. **System Environment Check**
   - Python version verification (3.8+)
   - Required package availability
   - System resources (memory, CPU, disk)
   - Working directory and project structure

2. **Data Uploads Verification**
   - Upload directory existence and permissions
   - File handling capabilities
   - Dataset connector availability

3. **Data Engineering Validation**
   - Trust scoring engine functionality
   - Database connectivity
   - ETL pipeline components
   - Data quality assessment tools

4. **LLM Engineering Check**
   - LLM lifecycle manager
   - Provider configurations
   - Model management systems

5. **High Performance System Test**
   - MoE system initialization
   - Expert ensemble functionality
   - Performance metrics collection
   - Domain routing systems

6. **Security System Verification**
   - Authentication manager
   - Security web UI
   - Authorization systems

7. **Service Connectivity**
   - MCP server status
   - Production server health
   - API endpoint availability

8. **Quality Assurance**
   - Test suite execution
   - Plugin system functionality
   - Dashboard availability

## 🔧 Problem Resolution

### Automatic Issue Detection

The system automatically detects common issues:

- **Missing Dependencies** - Package installation problems
- **Configuration Issues** - Missing or incorrect config files
- **Service Failures** - Server startup problems
- **Permission Issues** - File system access problems
- **Connectivity Problems** - Network and API issues

### Interactive Resolution

The problem resolver provides:

1. **Issue Identification** - Automatic problem detection
2. **Solution Database** - Pre-defined fixes for common issues
3. **Step-by-Step Guidance** - Clear resolution instructions
4. **Command Execution** - Automatic fix application
5. **Verification** - Post-fix validation

## 📈 System Status Monitoring

### Real-time Status Overview

```bash
# Get current system status
python workflow_launcher.py --status
```

**Monitors:**
- Production server health
- MCP server status
- File system integrity
- Python environment
- Component availability

### Component-Specific Checks

```bash
# Run detailed component checks
python workflow_launcher.py
# Then select option 8
```

**Checks:**
- Directory structure
- Key file availability
- Python file counts
- Component dependencies

## 🧪 Testing Integration

### Automated Test Execution

```bash
# Run all system tests
python workflow_launcher.py --tests
```

**Test Coverage:**
- Unit tests
- Integration tests
- Performance tests
- Component tests

### Test Results Analysis

- Pass/fail status
- Performance metrics
- Error details
- Recommendations

## 📊 Dashboard Management

### Available Dashboards

1. **Operation Sindoor Dashboard** - Specialized analysis dashboard
2. **Ultimate Analytics Dashboard** - Comprehensive system analytics
3. **SME Dashboard** - Subject Matter Expert interface
4. **Trust Scoring Dashboard** - Data quality and trust metrics

### Dashboard Launch

```bash
# Launch all dashboards
python launch_operation_sindoor_dashboard.py

# Or use the workflow launcher
python workflow_launcher.py
# Then select option 4
```

## 🔄 Complete Workflow Automation

### Full System Check & Fix

```bash
# Run complete automated workflow
python workflow_launcher.py --full
```

**Workflow Steps:**
1. **Diagnostic** - Comprehensive system check
2. **Problem Resolution** - Automatic issue fixing
3. **Testing** - Validation of fixes
4. **Service Start** - Production server launch
5. **Verification** - Final status check

## 📋 Report Management

### Diagnostic Reports

Reports are automatically generated and saved:
- `workflow_diagnostic_report_YYYYMMDD_HHMMSS.json`
- Comprehensive issue details
- Resolution recommendations
- Performance metrics

### Report Analysis

```bash
# View recent reports
python workflow_launcher.py
# Then select option 9
```

**Report Features:**
- Timestamp and duration
- Component status summary
- Issue details and recommendations
- Performance metrics

## 🛠️ Troubleshooting Guide

### Common Issues & Solutions

#### 1. Python Environment Issues
```bash
# Check Python version
python --version

# Install missing packages
pip install -r requirements.txt

# Upgrade pip
python -m pip install --upgrade pip
```

#### 2. Server Startup Problems
```bash
# Check if ports are in use
netstat -an | findstr 8003
netstat -an | findstr 8000

# Kill conflicting processes
taskkill /F /IM python.exe
```

#### 3. Permission Issues
```bash
# Create uploads directory
mkdir uploads

# Set permissions (Windows)
icacls uploads /grant Everyone:F

# Set permissions (Linux/Mac)
chmod 755 uploads
```

#### 4. Database Issues
```bash
# Check SQLite installation
python -c "import sqlite3; print('SQLite OK')"

# Create database directory
mkdir -p data
```

#### 5. Import Errors
```bash
# Check module availability
python -c "import high_performance_system.core.ultimate_moe_system; print('MoE OK')"

# Install missing dependencies
pip install numpy pandas fastapi uvicorn streamlit plotly
```

## 📊 Performance Monitoring

### Key Metrics

- **Response Time** - API endpoint latency
- **Throughput** - Requests per second
- **Error Rate** - Failed request percentage
- **Resource Usage** - CPU, memory, disk utilization

### Performance Optimization

1. **Caching** - Enable Redis caching
2. **Parallel Processing** - Use async operations
3. **Resource Management** - Monitor system resources
4. **Load Balancing** - Distribute requests

## 🔒 Security Considerations

### Security Checks

- **Authentication** - JWT token validation
- **Authorization** - Role-based access control
- **Data Encryption** - Sensitive data protection
- **Input Validation** - Request sanitization

### Security Best Practices

1. **Regular Updates** - Keep dependencies updated
2. **Access Control** - Implement proper permissions
3. **Monitoring** - Log security events
4. **Backup** - Regular data backups

## 📈 Advanced Usage

### Custom Diagnostics

```python
# Create custom diagnostic check
from complete_workflow_diagnostic import CompleteWorkflowDiagnostic

diagnostic = CompleteWorkflowDiagnostic()
report = await diagnostic.run_complete_diagnostic()
```

### Automated Monitoring

```python
# Set up automated monitoring
import schedule
import time

def run_daily_check():
    subprocess.run(["python", "workflow_launcher.py", "--diagnostic"])

schedule.every().day.at("09:00").do(run_daily_check)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Integration with CI/CD

```yaml
# GitHub Actions example
name: System Health Check
on: [push, pull_request]

jobs:
  diagnostic:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Diagnostic
        run: python workflow_launcher.py --diagnostic
      - name: Run Tests
        run: python workflow_launcher.py --tests
```

## 📞 Support & Maintenance

### Getting Help

1. **Run Diagnostic** - Identify the issue
2. **Check Logs** - Review error messages
3. **Use Resolver** - Apply automatic fixes
4. **Manual Resolution** - Follow step-by-step guides

### Maintenance Schedule

- **Daily** - Quick status check
- **Weekly** - Full diagnostic run
- **Monthly** - Performance review
- **Quarterly** - Security audit

## 🎯 Success Metrics

### System Health Indicators

- **Uptime** - 99.9% target
- **Response Time** - <100ms average
- **Error Rate** - <1% target
- **Test Coverage** - >90% target

### Performance Targets

- **Throughput** - 1000+ req/s
- **Latency** - <50ms p95
- **Memory Usage** - <2GB
- **CPU Usage** - <80%

## 📚 Additional Resources

### Documentation
- [OpenTrustEval Main README](README.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

### Tools
- [Operation Sindoor Dashboard](operation_sindoor_dashboard.py)
- [Production Server](superfast_production_server.py)
- [Test Suite](tests/)

### Support
- [Issue Tracker](https://github.com/Kumarvels/OpenTrustEval/issues)
- [Discussions](https://github.com/Kumarvels/OpenTrustEval/discussions)

---

## 🚀 Quick Reference

### Essential Commands
```bash
# Complete system check
python workflow_launcher.py --full

# Quick status
python workflow_launcher.py --status

# Start services
python workflow_launcher.py --server

# Run tests
python workflow_launcher.py --tests
```

### Key URLs
- **Production Server**: http://localhost:8003
- **MCP Server**: http://localhost:8000
- **Dashboards**: http://localhost:8501-8504

### Important Files
- **Diagnostic**: `complete_workflow_diagnostic.py`
- **Resolver**: `workflow_problem_resolver.py`
- **Launcher**: `workflow_launcher.py`
- **Server**: `superfast_production_server.py`

---

**🎯 This workflow system ensures your OpenTrustEval platform runs optimally with comprehensive monitoring, automated problem resolution, and step-by-step guidance for all system components.** 