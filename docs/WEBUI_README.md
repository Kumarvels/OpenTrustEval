# 🌐 OpenTrustEval Workflow Web UI System

## 🎯 **OVERVIEW**

The **OpenTrustEval Workflow Web UI System** provides a comprehensive, user-friendly web interface for managing all workflow solutions and automated scripts. This system integrates all diagnostic, problem resolution, and system management capabilities into a single, intuitive web application.

## 🚀 **QUICK START**

### **Launch the Web UI**
```bash
# Automatic launcher with dependency checking
python launch_workflow_webui.py

# Manual launch
streamlit run workflow_webui.py

# Direct access
http://localhost:8501
```

### **System Requirements**
- Python 3.8+
- Streamlit
- Plotly
- Pandas
- Requests
- Psutil

## 📋 **WEB UI FEATURES**

### 🏠 **Dashboard**
- **System Overview**: Real-time status of all components
- **Quick Actions**: One-click access to common operations
- **Component Status**: Visual status of all system components
- **Recent Activity**: Log of recent system activities

### 🔍 **System Diagnostic**
- **Complete Diagnostic**: Full system health check
- **Quick Diagnostic**: Fast component verification
- **Component-Specific**: Targeted component testing
- **Real-time Results**: Live diagnostic results display

### 🔧 **Problem Resolution**
- **Interactive Resolution**: Step-by-step problem fixing
- **Automatic Fix**: One-click issue resolution
- **Manual Steps**: Detailed resolution guidance
- **Resolution History**: Track of all fixes applied

### 🚀 **Service Management**
- **Production Server**: Start/stop and monitor main API server
- **MCP Server**: Model Context Protocol server management
- **Dashboard Launcher**: Launch specialized dashboards
- **Performance Monitoring**: Real-time service metrics

### 📊 **Analytics & Monitoring**
- **System Metrics**: CPU, memory, disk, network usage
- **Performance Analytics**: Request rates, response times, error rates
- **Component Analysis**: Health scores and file counts
- **Real-time Monitoring**: Live system monitoring

### 🧪 **Testing & Validation**
- **Unit Tests**: Individual component testing
- **Integration Tests**: System integration validation
- **Performance Tests**: Load and stress testing
- **End-to-End Tests**: Complete workflow validation

### 📋 **Reports & Logs**
- **Diagnostic Reports**: Detailed system health reports
- **System Logs**: Real-time log monitoring
- **Performance Reports**: System performance analysis
- **Report Management**: Historical report access

### ⚙️ **Configuration**
- **System Config**: Python environment and file system settings
- **Service Config**: Server configurations and endpoints
- **Dashboard Config**: Dashboard management and launching
- **Settings Management**: System-wide configuration

## 🎯 **INTEGRATED WORKFLOW SOLUTIONS**

### **Automated Scripts Integration**

The Web UI seamlessly integrates all workflow solutions:

#### **1. Diagnostic System**
- `complete_workflow_diagnostic.py` - 13-step comprehensive diagnostic
- Real-time component health monitoring
- Automatic issue detection and reporting

#### **2. Problem Resolution**
- `workflow_problem_resolver.py` - Interactive problem fixing
- Solution database with common fixes
- Step-by-step resolution guidance

#### **3. Service Management**
- `workflow_launcher.py` - Unified service management
- Production server control
- Dashboard launching

#### **4. Operation Sindoor Dashboard**
- `operation_sindoor_dashboard.py` - Specialized analysis dashboard
- Real-time misinformation detection
- Interactive visualizations

## 📊 **WEB UI ARCHITECTURE**

### **Frontend (Streamlit)**
```python
# Main interface structure
class WorkflowWebUI:
    def main_interface(self):
        # Navigation sidebar
        # Page routing
        # Real-time updates
    
    def dashboard_page(self):
        # System overview
        # Quick actions
        # Component status
    
    def diagnostic_page(self):
        # Diagnostic options
        # Results display
        # Recommendations
```

### **Backend Integration**
```python
# Script integration
def run_diagnostic_async(self):
    subprocess.run([sys.executable, self.diagnostic_script])

def run_problem_resolver(self):
    subprocess.run([sys.executable, self.resolver_script])

def start_production_server(self):
    subprocess.Popen([sys.executable, self.production_server])
```

### **Real-time Monitoring**
```python
# Status monitoring
def show_quick_status(self):
    # Production server health
    # MCP server status
    # File system status
    # Python environment
```

## 🎯 **USAGE GUIDELINES**

### **Getting Started**

1. **Launch Web UI**
   ```bash
   python launch_workflow_webui.py
   ```

2. **Access Dashboard**
   - Open browser to `http://localhost:8501`
   - Navigate using sidebar menu

3. **Run Initial Diagnostic**
   - Go to "🔍 System Diagnostic"
   - Click "🚀 Run Diagnostic"
   - Review results and recommendations

4. **Resolve Issues**
   - Go to "🔧 Problem Resolution"
   - Follow interactive resolution steps
   - Apply automatic fixes

### **Daily Operations**

#### **Morning Check**
1. **Dashboard Review**
   - Check system status
   - Review recent activity
   - Verify service health

2. **Quick Diagnostic**
   - Run quick system check
   - Address any issues
   - Start required services

#### **Monitoring**
1. **Analytics Page**
   - Monitor system performance
   - Track resource usage
   - Review component health

2. **Service Management**
   - Ensure all services running
   - Monitor performance metrics
   - Handle service issues

#### **Maintenance**
1. **Testing**
   - Run test suites
   - Validate system integrity
   - Check component functionality

2. **Reports**
   - Review diagnostic reports
   - Analyze performance data
   - Track system changes

## 🔧 **ADVANCED FEATURES**

### **Custom Dashboards**

The Web UI supports launching specialized dashboards:

```bash
# Operation Sindoor Dashboard
python launch_operation_sindoor_dashboard.py

# Ultimate Analytics Dashboard
streamlit run high_performance_system/analytics/ultimate_analytics_dashboard.py

# Trust Scoring Dashboard
streamlit run data_engineering/trust_scoring_dashboard.py
```

### **API Integration**

The Web UI integrates with all system APIs:

```python
# Production Server API
http://localhost:8003/health
http://localhost:8003/performance
http://localhost:8003/detect

# MCP Server API
http://localhost:8000/health
```

### **Real-time Updates**

The Web UI provides real-time system updates:

- **Live Status Monitoring**: Real-time component status
- **Performance Tracking**: Live performance metrics
- **Log Streaming**: Real-time log monitoring
- **Alert System**: Immediate issue notifications

## 📊 **MONITORING & ANALYTICS**

### **System Metrics**

The Web UI tracks comprehensive system metrics:

- **CPU Usage**: Real-time CPU utilization
- **Memory Usage**: Memory consumption tracking
- **Disk Usage**: Storage space monitoring
- **Network I/O**: Network activity tracking

### **Performance Analytics**

Advanced performance monitoring:

- **Request Rates**: Requests per second
- **Response Times**: Average response latency
- **Error Rates**: Error percentage tracking
- **Cache Performance**: Cache hit rates

### **Component Health**

Component-specific health monitoring:

- **Health Scores**: Component health ratings
- **File Counts**: Component file statistics
- **Dependency Status**: Dependency verification
- **Integration Status**: Component integration levels

## 🛠️ **TROUBLESHOOTING**

### **Common Issues**

#### **Web UI Won't Start**
```bash
# Check dependencies
pip install streamlit plotly pandas requests psutil

# Check port availability
netstat -an | findstr 8501

# Manual launch
streamlit run workflow_webui.py --server.port 8502
```

#### **Scripts Not Working**
```bash
# Check script permissions
chmod +x *.py

# Run with explicit Python
python complete_workflow_diagnostic.py

# Check error logs
tail -f workflow_webui.log
```

#### **Services Not Responding**
```bash
# Check service status
curl http://localhost:8003/health
curl http://localhost:8000/health

# Restart services
python workflow_launcher.py --server
```

### **Debug Mode**

Enable debug mode for detailed logging:

```bash
# Debug launch
streamlit run workflow_webui.py --logger.level debug

# Verbose output
python launch_workflow_webui.py --verbose
```

## 📈 **PERFORMANCE OPTIMIZATION**

### **Web UI Performance**

- **Caching**: Results caching for faster access
- **Async Operations**: Non-blocking script execution
- **Resource Monitoring**: Efficient resource usage
- **Connection Pooling**: Optimized API connections

### **System Integration**

- **Parallel Processing**: Concurrent script execution
- **Background Tasks**: Non-blocking operations
- **Resource Management**: Efficient resource allocation
- **Error Handling**: Robust error recovery

## 🔒 **SECURITY FEATURES**

### **Access Control**

- **Local Access**: Localhost-only access by default
- **Authentication**: Optional authentication system
- **Authorization**: Role-based access control
- **Audit Logging**: Comprehensive activity logging

### **Data Protection**

- **Secure Communication**: HTTPS support
- **Data Encryption**: Sensitive data encryption
- **Input Validation**: Secure input handling
- **Error Sanitization**: Safe error messages

## 📚 **API REFERENCE**

### **Web UI Endpoints**

```python
# Main interface
http://localhost:8501

# Dashboard pages
http://localhost:8501/?page=dashboard
http://localhost:8501/?page=diagnostic
http://localhost:8501/?page=resolution
http://localhost:8501/?page=services
http://localhost:8501/?page=analytics
http://localhost:8501/?page=testing
http://localhost:8501/?page=reports
http://localhost:8501/?page=configuration
```

### **Script Integration**

```python
# Diagnostic script
python complete_workflow_diagnostic.py

# Problem resolver
python workflow_problem_resolver.py

# Service launcher
python workflow_launcher.py --server

# Dashboard launcher
python launch_operation_sindoor_dashboard.py
```

## 🎯 **BEST PRACTICES**

### **Daily Operations**

1. **Start with Dashboard**
   - Review system status
   - Check for issues
   - Plan daily tasks

2. **Run Regular Diagnostics**
   - Daily quick checks
   - Weekly full diagnostics
   - Monthly comprehensive reviews

3. **Monitor Performance**
   - Track key metrics
   - Identify trends
   - Optimize performance

4. **Maintain System Health**
   - Address issues promptly
   - Update configurations
   - Backup important data

### **Troubleshooting Workflow**

1. **Identify Issue**
   - Use diagnostic tools
   - Check error logs
   - Review system status

2. **Apply Resolution**
   - Use automated fixes
   - Follow manual steps
   - Verify resolution

3. **Validate Fix**
   - Run tests
   - Check functionality
   - Monitor performance

4. **Document Changes**
   - Update logs
   - Record solutions
   - Share knowledge

## 🚀 **DEPLOYMENT**

### **Production Deployment**

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
export OPENTRUSTEVAL_ENV=production

# Launch with production settings
streamlit run workflow_webui.py --server.port 8501 --server.address 0.0.0.0
```

### **Docker Deployment**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "workflow_webui.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Cloud Deployment**

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opentrusteval-webui
spec:
  replicas: 1
  selector:
    matchLabels:
      app: opentrusteval-webui
  template:
    metadata:
      labels:
        app: opentrusteval-webui
    spec:
      containers:
      - name: webui
        image: opentrusteval/webui:latest
        ports:
        - containerPort: 8501
```

## 📞 **SUPPORT**

### **Getting Help**

1. **Check Documentation**
   - Review this README
   - Check component documentation
   - Review error logs

2. **Use Diagnostic Tools**
   - Run system diagnostic
   - Use problem resolver
   - Check system status

3. **Community Support**
   - GitHub issues
   - Discussion forums
   - Documentation wiki

### **Reporting Issues**

When reporting issues, include:

- **System Information**: OS, Python version, dependencies
- **Error Messages**: Complete error logs
- **Steps to Reproduce**: Detailed reproduction steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens

---

## 🏆 **CONCLUSION**

The **OpenTrustEval Workflow Web UI System** provides a comprehensive, user-friendly interface for managing all workflow solutions and automated scripts. With its intuitive design, real-time monitoring, and integrated problem resolution, it makes system management easy and efficient.

**Key Benefits:**
- ✅ **Unified Interface**: Single web UI for all operations
- ✅ **Real-time Monitoring**: Live system status and performance
- ✅ **Automated Resolution**: One-click problem fixing
- ✅ **Comprehensive Analytics**: Detailed system insights
- ✅ **Easy Integration**: Seamless script integration
- ✅ **User-friendly**: Intuitive web interface

**Start using the Web UI today to streamline your OpenTrustEval workflow management!**

---

**🎯 Access your OpenTrustEval Workflow Web UI at: http://localhost:8501** 