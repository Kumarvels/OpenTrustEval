# 🚀 OpenTrustEval Unified Workflow Web UI

## 📋 Overview

The **OpenTrustEval Unified Workflow Web UI** is a comprehensive web interface that integrates all system components into a single, powerful dashboard. This unified approach eliminates the need for multiple separate WebUIs and provides a seamless user experience.

## 🎯 Key Features

### **🏠 Dashboard**
- **System Overview**: Real-time status of all components
- **Quick Actions**: One-click access to common operations
- **Performance Metrics**: Live monitoring of system health
- **Recent Activity**: Timeline of system events

### **📁 Dataset Management** *(Integrated from Dataset WebUI)*
- **📤 Create Dataset**: Upload and import datasets (CSV, JSON, Parquet, Excel)
- **📋 List Datasets**: View all available datasets with metadata
- **🔍 Validate Dataset**: Run validation rules and quality checks
- **📊 Visualize Dataset**: Create interactive charts and graphs
- **📁 Export Dataset**: Export in multiple formats

### **🤖 LLM Model Manager** *(Integrated from LLM WebUI)*
- **📋 List Models**: View all available LLM models
- **➕ Add Model**: Add new models with different providers
- **🔧 Fine-tune Model**: Configure and train models
- **📊 Evaluate Model**: Assess model performance

### **🔒 Security Management** *(Integrated from Security WebUI)*
- **👥 User Management**: Create, list, and manage users
- **🔐 Authentication**: OAuth and SAML provider configuration
- **🛡️ Security Monitoring**: Real-time security alerts and metrics
- **📦 Dependency Scanner**: Vulnerability scanning and compliance

### **🔍 System Diagnostic**
- **Comprehensive Health Checks**: All system components
- **Environment Validation**: Dependencies and configuration
- **Performance Analysis**: System metrics and bottlenecks
- **Issue Detection**: Automated problem identification

### **🔧 Problem Resolution**
- **Interactive Fixes**: One-click problem resolution
- **Automated Troubleshooting**: Guided issue resolution
- **System Recovery**: Automated recovery procedures
- **Manual Overrides**: Advanced user controls

### **🚀 Service Management**
- **Start/Stop Services**: Control all system services
- **Service Monitoring**: Real-time service status
- **Configuration Management**: Service settings
- **Health Checks**: Automated service validation

### **📊 Analytics & Monitoring**
- **Performance Tracking**: Real-time metrics
- **Usage Analytics**: System utilization patterns
- **Error Monitoring**: Exception tracking and analysis
- **Custom Dashboards**: Configurable visualizations

### **🧪 Testing & Validation**
- **Automated Test Suites**: Comprehensive testing
- **Performance Testing**: Load and stress testing
- **Integration Testing**: Cross-component validation
- **Custom Test Creation**: User-defined test scenarios

### **📋 Reports & Logs**
- **System Reports**: Comprehensive system analysis
- **Error Logs**: Detailed error tracking
- **Performance Reports**: Historical performance data
- **Export Capabilities**: Multiple report formats

### **⚙️ Configuration**
- **System Settings**: Global configuration management
- **Component Settings**: Individual component configuration
- **Environment Variables**: System environment management
- **Backup & Restore**: Configuration backup capabilities

## 🚀 Quick Start

### **1. Launch the Unified Web UI**
```bash
python launch_workflow_webui.py
```

### **2. Access the Interface**
Open your browser and navigate to:
```
http://localhost:8501
```

### **3. Navigate Using the Sidebar**
Use the sidebar navigation to access different features:
- **🏠 Dashboard**: System overview
- **📁 Dataset Management**: Data operations
- **🤖 LLM Model Manager**: Model operations
- **🔒 Security Management**: Security operations
- **🔍 System Diagnostic**: Health checks
- **🔧 Problem Resolution**: Issue fixing
- **🚀 Service Management**: Service control
- **📊 Analytics & Monitoring**: Performance tracking
- **🧪 Testing & Validation**: Test execution
- **📋 Reports & Logs**: System reports
- **⚙️ Configuration**: Settings management

## 📊 System Requirements

### **Software Requirements**
- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Requests
- Psutil

### **Hardware Requirements**
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 10GB free space

### **Network Requirements**
- **Port 8501**: Web UI access
- **Port 8003**: Production server (optional)
- **Port 8000**: MCP server (optional)

## 🔧 Installation

### **1. Install Dependencies**
```bash
pip install streamlit pandas plotly requests psutil
```

### **2. Verify Installation**
```bash
python test_unified_webui.py
```

### **3. Launch the Web UI**
```bash
python launch_workflow_webui.py
```

## 📁 File Structure

```
OpenTrustEval/
├── workflow_webui.py              # Main unified web UI
├── launch_workflow_webui.py       # Web UI launcher
├── test_unified_webui.py          # Test suite
├── README_UNIFIED_WEBUI.md        # This file
├── data_engineering/              # Dataset management backend
├── llm_engineering/               # LLM management backend
├── security/                      # Security management backend
├── high_performance_system/       # High-performance components
├── mcp_server/                    # MCP server components
├── plugins/                       # Plugin system
└── tests/                         # Test suites
```

## 🎯 Usage Examples

### **Dataset Management**
1. Navigate to **📁 Dataset Management**
2. Upload a CSV file in the **📤 Create Dataset** tab
3. View your datasets in the **📋 List Datasets** tab
4. Validate data quality in the **🔍 Validate Dataset** tab
5. Create visualizations in the **📊 Visualize Dataset** tab

### **LLM Model Management**
1. Navigate to **🤖 LLM Model Manager**
2. Add a new model in the **➕ Add Model** tab
3. Configure fine-tuning in the **🔧 Fine-tune Model** tab
4. Evaluate performance in the **📊 Evaluate Model** tab

### **Security Management**
1. Navigate to **🔒 Security Management**
2. Create users in the **👥 User Management** tab
3. Configure authentication in the **🔐 Authentication** tab
4. Monitor security in the **🛡️ Security Monitoring** tab

### **System Diagnostics**
1. Navigate to **🔍 System Diagnostic**
2. Click **Run Comprehensive Diagnostic**
3. Review results and recommendations
4. Use **🔧 Problem Resolution** to fix any issues

## 🔍 Troubleshooting

### **Common Issues**

#### **Web UI Not Starting**
```bash
# Check if port 8501 is available
netstat -an | findstr :8501

# Kill any existing processes
taskkill /f /im python.exe

# Restart the web UI
python launch_workflow_webui.py
```

#### **Component Not Available**
```bash
# Run the test suite
python test_unified_webui.py

# Check component availability
python -c "import data_engineering.dataset_integration; print('Dataset Manager OK')"
```

#### **Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### **Performance Issues**
- **High Memory Usage**: Restart the web UI
- **Slow Response**: Check system resources
- **Connection Errors**: Verify network connectivity

## 📈 Performance Optimization

### **System Optimization**
- **Memory Management**: Monitor memory usage
- **CPU Optimization**: Balance workload distribution
- **Storage Optimization**: Regular cleanup of temporary files

### **Web UI Optimization**
- **Caching**: Enable result caching for repeated operations
- **Async Operations**: Use background processing for long tasks
- **Resource Limits**: Set appropriate limits for large datasets

## 🔒 Security Considerations

### **Access Control**
- **User Authentication**: Implement proper user management
- **Role-Based Access**: Configure appropriate permissions
- **Session Management**: Secure session handling

### **Data Security**
- **Data Encryption**: Encrypt sensitive data
- **Secure Uploads**: Validate uploaded files
- **Access Logging**: Monitor system access

## 📊 Monitoring & Analytics

### **System Monitoring**
- **Real-time Metrics**: Live performance tracking
- **Alert System**: Automated notifications
- **Historical Data**: Performance trends analysis

### **Usage Analytics**
- **Feature Usage**: Track most-used features
- **User Behavior**: Analyze user patterns
- **Performance Metrics**: System efficiency tracking

## 🔄 Updates & Maintenance

### **Regular Maintenance**
- **Log Rotation**: Manage log file sizes
- **Cache Cleanup**: Clear temporary files
- **Database Optimization**: Optimize data storage

### **System Updates**
- **Dependency Updates**: Keep packages current
- **Security Patches**: Apply security updates
- **Feature Updates**: Deploy new features

## 📞 Support

### **Documentation**
- **User Guide**: Comprehensive usage instructions
- **API Documentation**: Technical reference
- **Troubleshooting Guide**: Common issues and solutions

### **Community Support**
- **GitHub Issues**: Report bugs and request features
- **Discussion Forums**: Community discussions
- **Email Support**: Direct support contact

## 🏆 Benefits of Unified Approach

### **🎯 Single Interface**
- **No Multiple Tabs**: Everything in one place
- **Consistent UI**: Unified design language
- **Seamless Navigation**: Easy switching between features

### **⚡ Performance**
- **Reduced Overhead**: Single web server
- **Shared Resources**: Efficient resource utilization
- **Faster Loading**: Optimized component loading

### **🔧 Maintenance**
- **Centralized Management**: Single codebase to maintain
- **Unified Updates**: Synchronized feature updates
- **Simplified Deployment**: Single deployment process

### **📊 Analytics**
- **Unified Metrics**: Comprehensive system analytics
- **Cross-Component Insights**: Integrated performance data
- **Better Monitoring**: Holistic system view

## 🚀 Future Enhancements

### **Planned Features**
- **Advanced Analytics**: Machine learning insights
- **Custom Dashboards**: User-configurable views
- **API Integration**: External system integration
- **Mobile Support**: Responsive mobile interface

### **Performance Improvements**
- **Real-time Updates**: Live data streaming
- **Advanced Caching**: Intelligent caching strategies
- **Load Balancing**: Distributed processing

---

## 🎉 Conclusion

The **OpenTrustEval Unified Workflow Web UI** provides a powerful, integrated solution for managing all aspects of the OpenTrustEval system. With its comprehensive feature set, intuitive interface, and robust architecture, it offers an unparalleled user experience for system management and monitoring.

**Access your unified interface at: http://localhost:8501**

---

*For more information, visit the main OpenTrustEval documentation or contact the development team.* 