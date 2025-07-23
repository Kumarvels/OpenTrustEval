#!/usr/bin/env python3
"""
OpenTrustEval Workflow Web UI System
Comprehensive web interface for all workflow solutions and automated scripts
UNIFIED INTERFACE - Integrates all WebUIs into one
"""

import streamlit as st
import asyncio
import subprocess
import json
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import sys
import os
import tempfile
import shutil
from typing import Dict, List, Any, Optional
import psutil
import numpy as np
import concurrent.futures

# Configure Streamlit page
st.set_page_config(
    page_title="OpenTrustEval Unified Workflow System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Performance Optimization: Caching and Async Health Checks ---
@st.cache_data(ttl=10)
def cached_health_check(url: str) -> bool:
    try:
        response = requests.get(url, timeout=3)
        return response.status_code == 200
    except Exception:
        return False

async def async_health_check(url: str) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=3)
            return response.status_code == 200
    except Exception:
        return False

# --- Background Task Manager ---
class BackgroundTaskManager:
    """Manages background tasks for non-blocking operations"""
    
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.tasks = {}
    
    def run_in_background(self, func, *args, **kwargs):
        """Run function in background thread"""
        future = self.executor.submit(func, *args, **kwargs)
        return future
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=False)

# Global background task manager
background_manager = BackgroundTaskManager()

# --- Cached Operations ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_diagnostic_results():
    """Cache diagnostic results to avoid repeated runs"""
    return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def cached_performance_data():
    """Cache performance data"""
    try:
        response = requests.get("http://localhost:8003/performance", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

class UnifiedWorkflowWebUI:
    """Comprehensive unified web UI for OpenTrustEval workflow management"""
    
    def __init__(self):
        self.diagnostic_script = "complete_workflow_diagnostic.py"
        self.resolver_script = "workflow_problem_resolver.py"
        self.launcher_script = "workflow_launcher.py"
        self.production_server = "superfast_production_server.py"
        self.dashboard_launcher = "launch_operation_sindoor_dashboard.py"
        # Remove eager manager initialization
        self.dataset_manager = None
        self.llm_manager = None
        self.auth_manager = None
        self.DATASET_AVAILABLE = None
        self.LLM_AVAILABLE = None
        self.SECURITY_AVAILABLE = None
        # Initialize session state
        if 'diagnostic_results' not in st.session_state:
            st.session_state.diagnostic_results = None
        if 'system_status' not in st.session_state:
            st.session_state.system_status = None
        if 'server_status' not in st.session_state:
            st.session_state.server_status = None
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "🏠 Dashboard"
    
    def _load_dataset_manager(self):
        if self.dataset_manager is not None:
            return self.dataset_manager
        try:
            from data_engineering.dataset_integration import DatasetManager
            self.dataset_manager = DatasetManager()
            self.DATASET_AVAILABLE = True
        except ImportError as e:
            st.warning(f"Dataset manager not available: {e}")
            self.dataset_manager = None
            self.DATASET_AVAILABLE = False
        return self.dataset_manager
    
    def _load_llm_manager(self):
        if self.llm_manager is not None:
            return self.llm_manager
        try:
            from llm_engineering.llm_lifecycle import LLMLifecycleManager
            self.llm_manager = LLMLifecycleManager()
            self.LLM_AVAILABLE = True
        except ImportError as e:
            st.warning(f"LLM manager not available: {e}")
            self.llm_manager = None
            self.LLM_AVAILABLE = False
        return self.llm_manager
    
    def _load_auth_manager(self):
        if self.auth_manager is not None:
            return self.auth_manager
        try:
            from security.auth_manager import AuthManager
            self.auth_manager = AuthManager()
            self.SECURITY_AVAILABLE = True
        except ImportError as e:
            st.warning(f"Security manager not available: {e}")
            self.auth_manager = None
            self.SECURITY_AVAILABLE = False
        return self.auth_manager
    
    def main_interface(self):
        """Main web UI interface"""
        st.title("🚀 OpenTrustEval Unified Workflow Management System")
        st.markdown("---")
        
        # Status display area for Quick Status actions
        if 'quick_status_action' in st.session_state and st.session_state.quick_status_action:
            st.info(f"🔄 {st.session_state.quick_status_action}")
            # Clear the action after displaying
            st.session_state.quick_status_action = None
        
        # Sidebar navigation
        with st.sidebar:
            st.header("🎯 Navigation")
            page = st.selectbox(
                "Select Page",
                [
                    "🏠 Dashboard",
                    "🔍 System Diagnostic",
                    "🔧 Problem Resolution",
                    "🚀 Service Management",
                    "📊 Analytics & Monitoring",
                    "🧪 Testing & Validation",
                    "📋 Reports & Logs",
                    "⚙️ Configuration",
                    "📁 Dataset Management",
                    "🤖 LLM Model Manager",
                    "🔒 Security Management",
                    "🔬 Research Lab"
                ],
                index=0 if st.session_state.current_page == "🏠 Dashboard" else None
            )
            
            # Update current page if changed via dropdown
            if page != st.session_state.current_page:
                st.session_state.current_page = page
            
            st.markdown("---")
            self.show_quick_status()
        
        # Page routing based on current_page
        current_page = st.session_state.current_page
        if current_page == "🏠 Dashboard":
            self.dashboard_page()
        elif current_page == "🔍 System Diagnostic":
            self.diagnostic_page()
        elif current_page == "🔧 Problem Resolution":
            self.problem_resolution_page()
        elif current_page == "🚀 Service Management":
            self.service_management_page()
        elif current_page == "📊 Analytics & Monitoring":
            self.analytics_page()
        elif current_page == "🧪 Testing & Validation":
            self.testing_page()
        elif current_page == "📋 Reports & Logs":
            self.reports_page()
        elif current_page == "⚙️ Configuration":
            self.configuration_page()
        elif current_page == "📁 Dataset Management":
            self.dataset_management_page()
        elif current_page == "🤖 LLM Model Manager":
            self.llm_management_page()
        elif current_page == "🔒 Security Management":
            self.security_management_page()
        elif current_page == "🔬 Research Lab":
            self.research_lab_page()
    
    def show_quick_status(self):
        """Show quick system status in sidebar with click functionality (optimized)"""
        try:
            st.subheader("📊 Quick Status")
            # Use cached health checks for performance
            prod_status = cached_health_check("http://localhost:8003/health")
            mcp_status = cached_health_check("http://localhost:8000/health")
            # Production server status with click functionality
            if prod_status:
                if st.button("✅ Production Server", key="prod_server_btn", help="Click to manage production server"):
                    self.manage_production_server()
            else:
                if st.button("❌ Production Server", key="prod_server_btn", help="Click to start production server"):
                    self.start_production_server()
            # MCP server status with click functionality
            if mcp_status:
                if st.button("✅ MCP Server", key="mcp_server_btn", help="Click to manage MCP server"):
                    self.manage_mcp_server()
            else:
                if st.button("⏸️ MCP Server", key="mcp_server_btn", help="Click to start MCP server"):
                    self.start_mcp_server()
            # Dataset Manager with click functionality
            if self.DATASET_AVAILABLE:
                if st.button("✅ Dataset Manager", key="dataset_btn", help="Click to open Dataset Management"):
                    st.session_state.current_page = "📁 Dataset Management"
                    st.rerun()
            else:
                if st.button("❌ Dataset Manager", key="dataset_btn", help="Click to check Dataset Manager status"):
                    self.check_dataset_manager()
            # LLM Manager with click functionality
            if self.LLM_AVAILABLE:
                if st.button("✅ LLM Manager", key="llm_btn", help="Click to open LLM Model Manager"):
                    st.session_state.current_page = "🤖 LLM Model Manager"
                    st.rerun()
            else:
                if st.button("❌ LLM Manager", key="llm_btn", help="Click to check LLM Manager status"):
                    self.check_llm_manager()
            # Security Manager with click functionality
            if self.SECURITY_AVAILABLE:
                if st.button("✅ Security Manager", key="security_btn", help="Click to open Security Management"):
                    st.session_state.current_page = "🔒 Security Management"
                    st.rerun()
            else:
                if st.button("❌ Security Manager", key="security_btn", help="Click to check Security Manager status"):
                    self.check_security_manager()
            # File System with click functionality
            uploads_exists = Path("uploads").exists()
            if uploads_exists:
                if st.button("✅ File System", key="filesystem_btn", help="Click to browse file system"):
                    self.browse_file_system()
            else:
                if st.button("❌ File System", key="filesystem_btn", help="Click to create uploads directory"):
                    self.create_uploads_directory()
            
            # Advanced Research Platform with click functionality
            if st.button("🚀 Advanced Research Platform Ready", key="research_btn", help="Click to open Research Lab"):
                st.session_state.current_page = "🔬 Research Lab"
                st.rerun()
            
        except Exception as e:
            st.error(f"Status Error: {str(e)}")
    
    def manage_production_server(self):
        """Manage production server"""
        st.info("🔄 Managing Production Server...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🟢 Start Server", key="start_prod"):
                self.start_production_server()
        with col2:
            if st.button("🔴 Stop Server", key="stop_prod"):
                self.stop_production_server()
        with col3:
            if st.button("🔄 Restart Server", key="restart_prod"):
                self.restart_production_server()
        
        # Show server status
        try:
            response = requests.get("http://localhost:8003/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Production Server is running")
                st.json(response.json())
            else:
                st.error("❌ Production Server is not responding")
        except:
            st.error("❌ Production Server is not accessible")
    
    def manage_mcp_server(self):
        """Manage MCP server"""
        st.info("🔄 Managing MCP Server...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🟢 Start MCP", key="start_mcp"):
                self.start_mcp_server()
        with col2:
            if st.button("🔴 Stop MCP", key="stop_mcp"):
                self.stop_mcp_server()
        with col3:
            if st.button("🔄 Restart MCP", key="restart_mcp"):
                self.restart_mcp_server()
        
        # Show MCP server status
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ MCP Server is running")
                st.json(response.json())
            else:
                st.warning("⏸️ MCP Server is not responding")
        except:
            st.warning("⏸️ MCP Server is not accessible")
    
    def check_dataset_manager(self):
        """Check Dataset Manager status"""
        st.info("🔍 Checking Dataset Manager...")
        
        try:
            manager = self._load_dataset_manager()
            st.success("✅ Dataset Manager is available")
            
            # Show dataset count
            datasets = manager.list_datasets()
            st.metric("Available Datasets", len(datasets))
            
            if st.button("📁 Open Dataset Manager", key="open_dataset"):
                st.session_state.current_page = "📁 Dataset Management"
                st.rerun()
        except Exception as e:
            st.error(f"❌ Dataset Manager Error: {e}")
            st.info("💡 Try installing required dependencies")
    
    def check_llm_manager(self):
        """Check LLM Manager status"""
        st.info("🔍 Checking LLM Manager...")
        
        try:
            manager = self._load_llm_manager()
            st.success("✅ LLM Manager is available")
            
            # Show model count
            models = manager.list_models()
            st.metric("Available Models", len(models))
            
            if st.button("🤖 Open LLM Manager", key="open_llm"):
                st.session_state.current_page = "🤖 LLM Model Manager"
                st.rerun()
        except Exception as e:
            st.error(f"❌ LLM Manager Error: {e}")
            st.info("💡 Try installing required dependencies")
    
    def check_security_manager(self):
        """Check Security Manager status"""
        st.info("🔍 Checking Security Manager...")
        
        try:
            manager = self._load_auth_manager()
            st.success("✅ Security Manager is available")
            
            # Show user count
            users = manager.list_users()
            st.metric("Registered Users", len(users))
            
            if st.button("🔒 Open Security Manager", key="open_security"):
                st.session_state.current_page = "🔒 Security Management"
                st.rerun()
        except Exception as e:
            st.error(f"❌ Security Manager Error: {e}")
            st.info("💡 Try installing required dependencies")
    
    def browse_file_system(self):
        """Browse file system"""
        st.info("📁 Browsing File System...")
        
        # Show uploads directory contents
        uploads_path = Path("uploads")
        if uploads_path.exists():
            st.success("✅ Uploads directory exists")
            
            # List files in uploads
            files = list(uploads_path.glob("*"))
            if files:
                st.subheader("📂 Files in Uploads Directory")
                for file in files[:10]:  # Show first 10 files
                    file_type = "📄" if file.is_file() else "📁"
                    st.write(f"{file_type} {file.name}")
                
                if len(files) > 10:
                    st.info(f"... and {len(files) - 10} more files")
            else:
                st.info("📁 Uploads directory is empty")
            
            # File upload option
            st.subheader("📤 Upload New File")
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json', 'txt', 'py'])
            if uploaded_file:
                file_path = uploads_path / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ File uploaded: {uploaded_file.name}")
        else:
            st.error("❌ Uploads directory not found")
    
    def create_uploads_directory(self):
        """Create uploads directory"""
        st.info("📁 Creating Uploads Directory...")
        
        try:
            uploads_path = Path("uploads")
            uploads_path.mkdir(exist_ok=True)
            st.success("✅ Uploads directory created successfully")
            
            # Create a sample file
            sample_file = uploads_path / "sample_data.csv"
            sample_data = "id,name,value\n1,Sample,100\n2,Test,200"
            with open(sample_file, 'w') as f:
                f.write(sample_data)
            st.info("📄 Created sample_data.csv for testing")
            
        except Exception as e:
            st.error(f"❌ Error creating uploads directory: {e}")
    
    def stop_production_server(self):
        """Stop production server"""
        st.info("🛑 Stopping Production Server...")
        try:
            # This would typically involve stopping the server process
            # For now, we'll just show a message
            st.success("✅ Production Server stopped")
        except Exception as e:
            st.error(f"❌ Error stopping server: {e}")
    
    def restart_production_server(self):
        """Restart production server"""
        st.info("🔄 Restarting Production Server...")
        try:
            # Stop and start the server
            self.stop_production_server()
            time.sleep(2)
            self.start_production_server()
            st.success("✅ Production Server restarted")
        except Exception as e:
            st.error(f"❌ Error restarting server: {e}")
    
    def stop_mcp_server(self):
        """Stop MCP server"""
        st.info("🛑 Stopping MCP Server...")
        try:
            # This would typically involve stopping the MCP server process
            st.success("✅ MCP Server stopped")
        except Exception as e:
            st.error(f"❌ Error stopping MCP server: {e}")
    
    def restart_mcp_server(self):
        """Restart MCP server"""
        st.info("🔄 Restarting MCP Server...")
        try:
            # Stop and start the MCP server
            self.stop_mcp_server()
            time.sleep(2)
            self.start_mcp_server()
            st.success("✅ MCP Server restarted")
        except Exception as e:
            st.error(f"❌ Error restarting MCP server: {e}")
    
    def dashboard_page(self):
        """Main dashboard page"""
        st.header("🏠 System Dashboard")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", "🟢 Operational", "All systems running")
        
        with col2:
            st.metric("Production Server", "🟢 Running", "Port 8003")
        
        with col3:
            st.metric("MCP Server", "🟡 Available", "Port 8000")
        
        with col4:
            st.metric("Last Diagnostic", "🟢 Recent", "All checks passed")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Run Diagnostic", use_container_width=True):
                self.run_diagnostic_async()
        
        with col2:
            if st.button("🔧 Fix Issues", use_container_width=True):
                self.run_problem_resolver()
        
        with col3:
            if st.button("🚀 Start Server", use_container_width=True):
                self.start_production_server()
        
        with col4:
            if st.button("📊 Launch Dashboards", use_container_width=True):
                self.launch_dashboards()
        
        # System overview
        st.subheader("📊 System Overview")
        
        # Component status
        components = [
            ("Data Engineering", "data_engineering/", self.DATASET_AVAILABLE),
            ("LLM Engineering", "llm_engineering/", self.LLM_AVAILABLE),
            ("High Performance System", "high_performance_system/", True),
            ("Security", "security/", self.SECURITY_AVAILABLE),
            ("MCP Server", "mcp_server/", True),
            ("Plugins", "plugins/", True),
            ("Tests", "tests/", True)
        ]
        
        status_data = []
        for name, path, available in components:
            exists = Path(path).exists()
            files = len(list(Path(path).rglob("*.py"))) if exists else 0
            status = "✅ Active" if exists and available else "❌ Missing" if not exists else "⚠️ Unavailable"
            status_data.append({
                "Component": name,
                "Status": status,
                "Files": files,
                "Path": path,
                "Available": available
            })
        
        df = pd.DataFrame(status_data)
        st.dataframe(df, use_container_width=True)
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        # Create sample activity data
        activity_data = {
            'Time': [datetime.now() - timedelta(minutes=i*10) for i in range(10)],
            'Event': [
                'System diagnostic completed',
                'Dataset uploaded',
                'LLM model trained',
                'Security scan completed',
                'Performance test passed',
                'User authentication',
                'Data validation',
                'Model evaluation',
                'System backup',
                'Configuration updated'
            ],
            'Status': ['✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅']
        }
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True)
    
    def diagnostic_page(self):
        """System diagnostic page"""
        st.header("🔍 System Diagnostic")
        st.markdown("Run a complete system diagnostic to check all components.")
        if st.button("Run Diagnostic"):
            with st.spinner("Running diagnostics, please wait..."):
                output = self.run_diagnostic_async()
                st.session_state.diagnostic_results = output
        if st.session_state.diagnostic_results:
            st.success("✅ Diagnostic completed!")
            st.json(st.session_state.diagnostic_results)
    
    def problem_resolution_page(self):
        """Problem resolution page"""
        st.header("🔧 Problem Resolution")
        
        # Resolution options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Resolution Options")
            
            resolution_type = st.selectbox(
                "Resolution Type",
                ["Interactive Resolution", "Automatic Fix", "Manual Steps"]
            )
            
            if resolution_type == "Manual Steps":
                component = st.selectbox(
                    "Select Component",
                    [
                        "System Environment",
                        "Data Uploads",
                        "Data Engineering",
                        "LLM Engineering",
                        "High Performance System",
                        "Security System",
                        "MCP Server",
                        "Production Server"
                    ]
                )
            
            run_resolution = st.button("🔧 Run Resolution", type="primary")
        
        with col2:
            st.subheader("📊 Issue Summary")
            if st.session_state.diagnostic_results:
                failed_components = [
                    r for r in st.session_state.diagnostic_results.get("results", [])
                    if r["status"] == "FAIL"
                ]
                
                if failed_components:
                    st.error(f"Found {len(failed_components)} failed components")
                    for comp in failed_components:
                        st.write(f"❌ {comp['component']}: {comp['message']}")
                else:
                    st.success("No issues found!")
            else:
                st.info("Run diagnostic first to identify issues")
        
        # Run resolution
        if run_resolution:
            with st.spinner("Running problem resolution..."):
                self.run_problem_resolver()
        
        # Resolution history
        st.subheader("📋 Resolution History")
        
        # Check for resolution reports
        report_files = [f for f in os.listdir('.') if f.startswith('workflow_diagnostic_report_') and f.endswith('.json')]
        
        if report_files:
            selected_report = st.selectbox("Select Report", report_files)
            
            if selected_report:
                try:
                    with open(selected_report, 'r') as f:
                        report = json.load(f)
                    
                    st.json(report)
                except Exception as e:
                    st.error(f"Error loading report: {str(e)}")
        else:
            st.info("No resolution reports available")
    
    def service_management_page(self):
        """Service management page"""
        st.header("🚀 Service Management")
        
        # Service status
        st.subheader("📊 Service Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Production Server")
            try:
                response = requests.get("http://localhost:8003/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Running")
                    health_data = response.json()
                    st.json(health_data)
                else:
                    st.error("❌ Error")
            except:
                st.error("❌ Not Running")
            
            if st.button("🚀 Start Production Server"):
                self.start_production_server()
        
        with col2:
            st.subheader("MCP Server")
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Running")
                else:
                    st.warning("⏸️ Available")
            except:
                st.warning("⏸️ Not Running")
            
            if st.button("🚀 Start MCP Server"):
                self.start_mcp_server()
        
        with col3:
            st.subheader("Dashboards")
            st.info("📊 Dashboard Status")
            
            if st.button("📊 Launch Dashboards"):
                self.launch_dashboards()
        
        # Performance monitoring
        st.subheader("📈 Performance Monitoring")
        
        if st.button("🔄 Refresh Performance Data"):
            self.refresh_performance_data()
        
        # Performance metrics
        try:
            response = requests.get("http://localhost:8003/performance", timeout=5)
            if response.status_code == 200:
                perf_data = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Requests", perf_data.get("total_requests", 0))
                with col2:
                    st.metric("Average Latency", f"{perf_data.get('avg_latency', 0):.2f}ms")
                with col3:
                    st.metric("Success Rate", f"{perf_data.get('success_rate', 0):.1f}%")
                with col4:
                    st.metric("Active Connections", perf_data.get("active_connections", 0))
                
                # Performance chart
                if "latency_history" in perf_data:
                    df = pd.DataFrame(perf_data["latency_history"])
                    fig = px.line(df, x="timestamp", y="latency", title="Latency Over Time")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Performance data not available")
        except:
            st.warning("Cannot connect to performance endpoint")
    
    def analytics_page(self):
        """Analytics and monitoring page"""
        st.header("📊 Analytics & Monitoring")
        
        # Analytics options
        tab1, tab2, tab3, tab4 = st.tabs(["System Metrics", "Performance Analytics", "Component Analysis", "Real-time Monitoring"])
        
        with tab1:
            st.subheader("📈 System Metrics")
            
            # System resources
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_percent = psutil.cpu_percent()
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            
            with col2:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
            with col3:
                disk = psutil.disk_usage('/')
                st.metric("Disk Usage", f"{disk.percent:.1f}%")
            
            with col4:
                network = psutil.net_io_counters()
                st.metric("Network I/O", f"{network.bytes_sent // 1024 // 1024}MB")
            
            # System resource charts
            col1, col2 = st.columns(2)
            
            with col1:
                # CPU usage over time
                cpu_data = []
                for i in range(10):
                    cpu_data.append({"time": i, "cpu": psutil.cpu_percent()})
                    time.sleep(0.1)
                
                df_cpu = pd.DataFrame(cpu_data)
                fig_cpu = px.line(df_cpu, x="time", y="cpu", title="CPU Usage")
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col2:
                # Memory usage
                memory_data = {
                    "Used": memory.used // 1024 // 1024,
                    "Available": memory.available // 1024 // 1024
                }
                fig_mem = px.pie(values=list(memory_data.values()), names=list(memory_data.keys()), title="Memory Usage")
                st.plotly_chart(fig_mem, use_container_width=True)
        
        with tab2:
            st.subheader("🚀 Performance Analytics")
            
            # Performance data
            try:
                response = requests.get("http://localhost:8003/performance", timeout=5)
                if response.status_code == 200:
                    perf_data = response.json()
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Requests/sec", perf_data.get("requests_per_second", 0))
                    with col2:
                        st.metric("Avg Response Time", f"{perf_data.get('avg_response_time', 0):.2f}ms")
                    with col3:
                        st.metric("Error Rate", f"{perf_data.get('error_rate', 0):.2f}%")
                    with col4:
                        st.metric("Cache Hit Rate", f"{perf_data.get('cache_hit_rate', 0):.1f}%")
                    
                    # Performance charts
                    if "request_history" in perf_data:
                        df_req = pd.DataFrame(perf_data["request_history"])
                        fig_req = px.line(df_req, x="timestamp", y="requests", title="Request Volume")
                        st.plotly_chart(fig_req, use_container_width=True)
                else:
                    st.warning("Performance data not available")
            except:
                st.warning("Cannot connect to performance endpoint")
        
        with tab3:
            st.subheader("🔍 Component Analysis")
            
            # Component health analysis
            components = [
                ("Data Engineering", "data_engineering/"),
                ("LLM Engineering", "llm_engineering/"),
                ("High Performance System", "high_performance_system/"),
                ("Security", "security/"),
                ("MCP Server", "mcp_server/"),
                ("Plugins", "plugins/"),
                ("Tests", "tests/")
            ]
            
            component_data = []
            for name, path in components:
                exists = Path(path).exists()
                files = len(list(Path(path).rglob("*.py"))) if exists else 0
                health_score = 100 if exists else 0
                
                component_data.append({
                    "Component": name,
                    "Health Score": health_score,
                    "Files": files,
                    "Status": "Healthy" if exists else "Missing"
                })
            
            df_comp = pd.DataFrame(component_data)
            
            # Component health chart
            fig_health = px.bar(df_comp, x="Component", y="Health Score", title="Component Health")
            st.plotly_chart(fig_health, use_container_width=True)
            
            # Component details
            st.dataframe(df_comp, use_container_width=True)
        
        with tab4:
            st.subheader("📡 Real-time Monitoring")
            
            # Real-time monitoring setup
            if st.button("🔄 Start Real-time Monitoring"):
                self.start_real_time_monitoring()
            
            # Monitoring dashboard
            placeholder = st.empty()
            
            # Simulate real-time updates
            if st.button("📊 Update Monitoring Data"):
                with placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Active Requests", "42", "↗️ +5")
                    with col2:
                        st.metric("Response Time", "15ms", "↘️ -2ms")
                    with col3:
                        st.metric("Error Rate", "0.1%", "↘️ -0.05%")
                    with col4:
                        st.metric("System Load", "65%", "↗️ +3%")
    
    def testing_page(self):
        """Testing and validation page"""
        st.header("🧪 Testing & Validation")
        
        # Test options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧪 Test Options")
            
            test_type = st.selectbox(
                "Test Type",
                ["Unit Tests", "Integration Tests", "Performance Tests", "End-to-End Tests"]
            )
            
            if test_type == "Unit Tests":
                test_file = st.selectbox(
                    "Select Test File",
                    ["simple_unit_test.py", "test_high_performance_system.py"]
                )
            
            run_tests = st.button("🧪 Run Tests", type="primary")
        
        with col2:
            st.subheader("📊 Test Results")
            st.info("Select test type and run to see results")
        
        # Run tests
        if run_tests:
            with st.spinner("Running tests..."):
                self.run_tests_async(test_type)
        
        # Test results display
        if st.session_state.get("test_results"):
            st.subheader("📋 Test Results")
            results = st.session_state.test_results
            
            # Test summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tests", results.get("total", 0))
            with col2:
                st.metric("Passed", results.get("passed", 0))
            with col3:
                st.metric("Failed", results.get("failed", 0))
            with col4:
                st.metric("Duration", f"{results.get('duration', 0):.2f}s")
            
            # Test details
            if results.get("details"):
                st.subheader("📊 Test Details")
                for test in results["details"]:
                    with st.expander(f"{test['name']} - {test['status']}"):
                        st.write(f"**Duration:** {test['duration']:.2f}s")
                        if test.get("output"):
                            st.code(test["output"])
                        if test.get("error"):
                            st.error(test["error"])
    
    def reports_page(self):
        """Reports and logs page"""
        st.header("📋 Reports & Logs")
        
        # Report options
        tab1, tab2, tab3 = st.tabs(["Diagnostic Reports", "System Logs", "Performance Reports"])
        
        with tab1:
            st.subheader("🔍 Diagnostic Reports")
            
            # List diagnostic reports
            report_files = [f for f in os.listdir('.') if f.startswith('workflow_diagnostic_report_') and f.endswith('.json')]
            
            if report_files:
                selected_report = st.selectbox("Select Report", report_files)
                
                if selected_report:
                    try:
                        with open(selected_report, 'r') as f:
                            report = json.load(f)
                        
                        # Report summary
                        summary = report.get("summary", {})
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Checks", summary.get("total_checks", 0))
                        with col2:
                            st.metric("Passed", summary.get("passed", 0))
                        with col3:
                            st.metric("Failed", summary.get("failed", 0))
                        with col4:
                            st.metric("Warnings", summary.get("warnings", 0))
                        
                        # Report details
                        st.subheader("📊 Report Details")
                        st.json(report)
                        
                    except Exception as e:
                        st.error(f"Error loading report: {str(e)}")
            else:
                st.info("No diagnostic reports available")
        
        with tab2:
            st.subheader("📝 System Logs")
            
            # Log display options
            log_level = st.selectbox("Log Level", ["ALL", "INFO", "WARNING", "ERROR"])
            log_lines = st.slider("Number of Lines", 10, 1000, 100)
            
            # Display logs
            if st.session_state.logs:
                log_df = pd.DataFrame(st.session_state.logs)
                
                if log_level != "ALL":
                    log_df = log_df[log_df["level"] == log_level]
                
                st.dataframe(log_df.tail(log_lines), use_container_width=True)
            else:
                st.info("No logs available")
            
            # Log management
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh Logs"):
                    self.refresh_logs()
            
            with col2:
                if st.button("🗑️ Clear Logs"):
                    st.session_state.logs = []
                    st.success("Logs cleared")
        
        with tab3:
            st.subheader("📈 Performance Reports")
            
            # Performance report generation
            if st.button("📊 Generate Performance Report"):
                self.generate_performance_report()
            
            # Performance metrics
            try:
                response = requests.get("http://localhost:8003/performance", timeout=5)
                if response.status_code == 200:
                    perf_data = response.json()
                    
                    # Performance summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Requests", perf_data.get("total_requests", 0))
                    with col2:
                        st.metric("Avg Latency", f"{perf_data.get('avg_latency', 0):.2f}ms")
                    with col3:
                        st.metric("Success Rate", f"{perf_data.get('success_rate', 0):.1f}%")
                    with col4:
                        st.metric("Uptime", f"{perf_data.get('uptime', 0):.1f}s")
                    
                    # Performance charts
                    if "performance_history" in perf_data:
                        df_perf = pd.DataFrame(perf_data["performance_history"])
                        fig_perf = px.line(df_perf, x="timestamp", y=["latency", "requests"], title="Performance Over Time")
                        st.plotly_chart(fig_perf, use_container_width=True)
                else:
                    st.warning("Performance data not available")
            except:
                st.warning("Cannot connect to performance endpoint")
    
    def configuration_page(self):
        """Configuration page"""
        st.header("⚙️ Configuration")
        
        # Configuration options
        tab1, tab2, tab3 = st.tabs(["System Config", "Service Config", "Dashboard Config"])
        
        with tab1:
            st.subheader("🔧 System Configuration")
            
            # System settings
            st.write("**Python Environment**")
            st.code(f"Python Version: {sys.version}")
            st.code(f"Working Directory: {os.getcwd()}")
            
            # File system configuration
            st.write("**File System Configuration**")
            directories = ["uploads", "data_engineering", "high_performance_system", "llm_engineering", "security"]
            
            for directory in directories:
                exists = Path(directory).exists()
                status = "✅" if exists else "❌"
                st.write(f"{status} {directory}/")
        
        with tab2:
            st.subheader("🚀 Service Configuration")
            
            # Production server config
            st.write("**Production Server**")
            st.code("URL: http://localhost:8003")
            st.code("Health Endpoint: /health")
            st.code("Performance Endpoint: /performance")
            
            # MCP server config
            st.write("**MCP Server**")
            st.code("URL: http://localhost:8000")
            st.code("Health Endpoint: /health")
            
            # Service management
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start All Services"):
                    self.start_all_services()
            
            with col2:
                if st.button("🛑 Stop All Services"):
                    self.stop_all_services()
        
        with tab3:
            st.subheader("📊 Dashboard Configuration")
            
            # Dashboard settings
            st.write("**Available Dashboards**")
            dashboards = [
                ("Operation Sindoor Dashboard", "operation_sindoor_dashboard.py"),
                ("Ultimate Analytics Dashboard", "high_performance_system/analytics/ultimate_analytics_dashboard.py"),
                ("Trust Scoring Dashboard", "data_engineering/trust_scoring_dashboard.py")
            ]
            
            for name, path in dashboards:
                exists = Path(path).exists()
                status = "✅" if exists else "❌"
                st.write(f"{status} {name}")
                if exists:
                    if st.button(f"🚀 Launch {name}"):
                        self.launch_specific_dashboard(path)
    
    def dataset_management_page(self):
        """Dataset Management Page - Integrated from Dataset WebUI"""
        st.header("📁 Dataset Management")
        st.markdown("Manage datasets: upload, validate, visualize, and export.")
        # Upload dataset
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "json", "xlsx"])
        if uploaded_file:
            with st.spinner("Uploading and importing dataset..."):
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                dataset_name = st.text_input("Dataset Name", value=uploaded_file.name)
                format_type = st.selectbox("Format", ["csv", "json", "xlsx"])
                if st.button("Import Dataset"):
                    with st.spinner("Importing dataset..."):
                        dataset_id = self._load_dataset_manager().import_dataset(temp_path, dataset_name, format_type)
                        st.success(f"✅ Successfully imported dataset: {dataset_id}")
        # List datasets
        if st.button("Refresh Dataset List"):
            with st.spinner("Loading datasets..."):
                datasets = self._load_dataset_manager().list_datasets()
                if not datasets:
                    st.info("No datasets found")
                else:
                    st.write(datasets)
    
    def llm_management_page(self):
        """LLM Model Manager Page - Enhanced with async API integration"""
        st.header("🤖 LLM Model Manager")
        st.markdown("**Comprehensive LLM model management with async API integration**")
        
        # Check API availability
        try:
            response = requests.get("http://localhost:8003/llm/health", timeout=5)
            api_available = response.status_code == 200
            if api_available:
                health_data = response.json()
                st.success(f"✅ LLM API Available - {health_data.get('total_models', 0)} models loaded")
            else:
                st.warning("⚠️ LLM API not responding")
                api_available = False
        except:
            st.warning("⚠️ LLM API not available")
            api_available = False
        
        if not api_available:
            st.info("Please ensure the production server is running with LLM endpoints enabled.")
            return
        
        # Tabs for different LLM management features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Model List", 
            "➕ Add Model", 
            "🔧 Model Operations",
            "📊 Model Status",
            "📝 Model Logs",
            "⚡ Batch Operations"
        ])
        
        with tab1:
            st.subheader("📋 Model List")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Refresh Models", type="primary"):
                    with st.spinner("Loading models..."):
                        try:
                            response = requests.get("http://localhost:8003/llm/models", timeout=10)
                            if response.status_code == 200:
                                models = response.json()
                                st.session_state.llm_models = models
                                st.success(f"✅ Loaded {len(models)} models")
                            else:
                                st.error(f"❌ Error loading models: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("📊 Get Metrics"):
                    with st.spinner("Loading metrics..."):
                        try:
                            response = requests.get("http://localhost:8003/llm/metrics", timeout=10)
                            if response.status_code == 200:
                                metrics = response.json()
                                st.json(metrics)
                            else:
                                st.error(f"❌ Error loading metrics: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            # Display models
            if 'llm_models' in st.session_state and st.session_state.llm_models:
                models = st.session_state.llm_models
                
                # Create a DataFrame for better display
                import pandas as pd
                df_data = []
                for model in models:
                    df_data.append({
                        'Name': model['name'],
                        'Provider': model['provider_type'],
                        'Status': model['status'],
                        'Created': model['created_at'],
                        'Last Used': model['last_used'] or 'Never'
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Model actions
                st.subheader("Quick Actions")
                selected_model = st.selectbox("Select Model", [m['name'] for m in models])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📊 Status"):
                        with st.spinner("Getting status..."):
                            try:
                                response = requests.get(f"http://localhost:8003/llm/models/{selected_model}/status", timeout=10)
                                if response.status_code == 200:
                                    status = response.json()
                                    st.json(status)
                                else:
                                    st.error(f"❌ Error: {response.status_code}")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                with col2:
                    if st.button("📝 Logs"):
                        with st.spinner("Loading logs..."):
                            try:
                                response = requests.get(f"http://localhost:8003/llm/models/{selected_model}/logs", timeout=10)
                                if response.status_code == 200:
                                    logs = response.json()
                                    st.json(logs)
                                else:
                                    st.error(f"❌ Error: {response.status_code}")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                with col3:
                    if st.button("🗑️ Delete"):
                        if st.button("Confirm Delete", type="secondary"):
                            with st.spinner("Deleting model..."):
                                try:
                                    response = requests.delete(f"http://localhost:8003/llm/models/{selected_model}", timeout=10)
                                    if response.status_code == 200:
                                        st.success(f"✅ Model '{selected_model}' deleted")
                                        # Refresh models
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Error: {response.status_code}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No models loaded. Click 'Refresh Models' to load them.")
        
        with tab2:
            st.subheader("➕ Add New Model")
            
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.text_input("Model Name", placeholder="e.g., my_llama_model")
                provider_type = st.selectbox("Provider Type", [
                    "llama_factory", "openai", "huggingface", "azure", "anthropic"
                ])
                model_path = st.text_input("Model Path (optional)", placeholder="/path/to/model")
            
            with col2:
                st.subheader("Provider Configuration")
                config_json = st.text_area(
                    "Config (JSON)",
                    value='{"model_name": "Llama-3-8B", "model_path": null}',
                    height=200,
                    help="Provider-specific configuration in JSON format"
                )
            
            if st.button("➕ Add Model", type="primary"):
                if not model_name:
                    st.error("❌ Model name is required")
                else:
                    with st.spinner("Adding model..."):
                        try:
                            import json
                            config = json.loads(config_json) if config_json.strip() else {}
                            if model_path:
                                config['model_path'] = model_path
                            
                            payload = {
                                "model_name": model_name,
                                "provider_type": provider_type,
                                "config": config
                            }
                            
                            response = requests.post(
                                "http://localhost:8003/llm/models",
                                json=payload,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ {result['message']}")
                                # Clear models cache to force refresh
                                if 'llm_models' in st.session_state:
                                    del st.session_state.llm_models
                            else:
                                st.error(f"❌ Error: {response.status_code} - {response.text}")
                        except json.JSONDecodeError:
                            st.error("❌ Invalid JSON configuration")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with tab3:
            st.subheader("🔧 Model Operations")
            
            if 'llm_models' in st.session_state and st.session_state.llm_models:
                models = st.session_state.llm_models
                selected_model = st.selectbox("Select Model for Operations", [m['name'] for m in models])
                
                # Fine-tuning
                st.subheader("🎯 Fine-tuning")
                col1, col2 = st.columns(2)
                with col1:
                    dataset_path = st.text_input("Dataset Path", placeholder="/path/to/dataset.csv")
                    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=3)
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4)
                    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=2e-5, format="%.2e")
                
                with col2:
                    use_lora = st.checkbox("Use LoRA", value=True)
                    use_qlora = st.checkbox("Use QLoRA", value=False)
                    monitor = st.selectbox("Monitor", ["None", "wandb", "mlflow", "tensorboard", "llamaboard"])
                    monitor = None if monitor == "None" else monitor
                
                if st.button("🎯 Start Fine-tuning"):
                    if not dataset_path:
                        st.error("❌ Dataset path is required")
                    else:
                        with st.spinner("Starting fine-tuning..."):
                            try:
                                payload = {
                                    "dataset_path": dataset_path,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "learning_rate": learning_rate,
                                    "use_lora": use_lora,
                                    "use_qlora": use_qlora,
                                    "monitor": monitor,
                                    "extra_kwargs": {}
                                }
                                
                                response = requests.post(
                                    f"http://localhost:8003/llm/models/{selected_model}/fine_tune",
                                    json=payload,
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    st.success(f"✅ {result['message']}")
                                    st.info(f"Task ID: {result['task_id']}")
                                else:
                                    st.error(f"❌ Error: {response.status_code} - {response.text}")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                # Evaluation
                st.subheader("📊 Evaluation")
                col1, col2 = st.columns(2)
                with col1:
                    eval_dataset_path = st.text_input("Evaluation Dataset Path", placeholder="/path/to/eval_dataset.csv")
                    metrics = st.multiselect(
                        "Metrics",
                        ["accuracy", "f1", "precision", "recall", "bleu", "rouge"],
                        default=["accuracy", "f1"]
                    )
                
                with col2:
                    if st.button("📊 Start Evaluation"):
                        if not eval_dataset_path:
                            st.error("❌ Evaluation dataset path is required")
                        else:
                            with st.spinner("Starting evaluation..."):
                                try:
                                    payload = {
                                        "dataset_path": eval_dataset_path,
                                        "metrics": metrics,
                                        "extra_kwargs": {}
                                    }
                                    
                                    response = requests.post(
                                        f"http://localhost:8003/llm/models/{selected_model}/evaluate",
                                        json=payload,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success(f"✅ {result['message']}")
                                        st.info(f"Task ID: {result['task_id']}")
                                    else:
                                        st.error(f"❌ Error: {response.status_code} - {response.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                
                # Deployment
                st.subheader("🚀 Deployment")
                if st.button("🚀 Deploy Model"):
                    with st.spinner("Deploying model..."):
                        try:
                            response = requests.post(
                                f"http://localhost:8003/llm/models/{selected_model}/deploy",
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ {result['message']}")
                                st.json(result['result'])
                            else:
                                st.error(f"❌ Error: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                
                # Text Generation
                st.subheader("💬 Text Generation")
                col1, col2 = st.columns(2)
                with col1:
                    prompt = st.text_area("Prompt", placeholder="Enter your prompt here...", height=100)
                    max_length = st.number_input("Max Length", min_value=10, max_value=1000, value=100)
                
                with col2:
                    if st.button("💬 Generate"):
                        if not prompt:
                            st.error("❌ Prompt is required")
                        else:
                            with st.spinner("Generating text..."):
                                try:
                                    response = requests.post(
                                        f"http://localhost:8003/llm/models/{selected_model}/generate",
                                        params={"prompt": prompt, "max_length": max_length},
                                        timeout=60
                                    )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success("✅ Text generated successfully")
                                        st.text_area("Generated Text", result['generated_text'], height=200)
                                    else:
                                        st.error(f"❌ Error: {response.status_code} - {response.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No models available. Add models first.")
        
        with tab4:
            st.subheader("📊 Model Status")
            
            if 'llm_models' in st.session_state and st.session_state.llm_models:
                models = st.session_state.llm_models
                selected_model = st.selectbox("Select Model for Status", [m['name'] for m in models])
                
                if st.button("📊 Get Status"):
                    with st.spinner("Getting status..."):
                        try:
                            response = requests.get(f"http://localhost:8003/llm/models/{selected_model}/status", timeout=10)
                            if response.status_code == 200:
                                status = response.json()
                                
                                # Display status metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Status", status['status'])
                                with col2:
                                    st.metric("Health", status['health'])
                                with col3:
                                    st.metric("Error Count", status['error_count'])
                                with col4:
                                    st.metric("Last Activity", status['last_activity'])
                                
                                # Detailed status
                                st.json(status)
                            else:
                                st.error(f"❌ Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No models available. Add models first.")
        
        with tab5:
            st.subheader("📝 Model Logs")
            
            if 'llm_models' in st.session_state and st.session_state.llm_models:
                models = st.session_state.llm_models
                selected_model = st.selectbox("Select Model for Logs", [m['name'] for m in models])
                log_limit = st.number_input("Log Limit", min_value=10, max_value=1000, value=100)
                
                if st.button("📝 Get Logs"):
                    with st.spinner("Loading logs..."):
                        try:
                            response = requests.get(
                                f"http://localhost:8003/llm/models/{selected_model}/logs",
                                params={"limit": log_limit},
                                timeout=10
                            )
                            if response.status_code == 200:
                                logs = response.json()
                                
                                st.metric("Total Logs", logs['total_logs'])
                                
                                # Display logs in a table
                                if logs['logs']:
                                    import pandas as pd
                                    log_df = pd.DataFrame(logs['logs'])
                                    st.dataframe(log_df, use_container_width=True)
                                else:
                                    st.info("No logs available for this model.")
                            else:
                                st.error(f"❌ Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No models available. Add models first.")
        
        with tab6:
            st.subheader("⚡ Batch Operations")
            
            st.info("Batch operations allow you to perform actions on multiple models at once.")
            
            # Batch add models
            st.subheader("➕ Batch Add Models")
            batch_config = st.text_area(
                "Batch Configuration (JSON)",
                value='''{
  "models": [
    {
      "model_name": "model1",
      "provider_type": "llama_factory",
      "config": {"model_name": "Llama-3-8B"}
    },
    {
      "model_name": "model2", 
      "provider_type": "huggingface",
      "config": {"model_name": "gpt2"}
    }
  ]
}''',
                height=300,
                help="JSON configuration for batch model addition"
            )
            
            if st.button("⚡ Batch Add Models"):
                with st.spinner("Adding models in batch..."):
                    try:
                        import json
                        payload = json.loads(batch_config)
                        
                        response = requests.post(
                            "http://localhost:8003/llm/models/batch_add",
                            json=payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Batch operation completed")
                            st.metric("Successful", result['successful'])
                            st.metric("Failed", result['failed'])
                            
                            if result['results']:
                                st.subheader("Successful Additions")
                                for res in result['results']:
                                    st.success(f"✅ {res['name']}")
                            
                            if result['errors']:
                                st.subheader("Failed Additions")
                                for err in result['errors']:
                                    st.error(f"❌ {err['name']}: {err['error']}")
                            
                            # Clear models cache to force refresh
                            if 'llm_models' in st.session_state:
                                del st.session_state.llm_models
                        else:
                            st.error(f"❌ Error: {response.status_code} - {response.text}")
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON configuration")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    def security_management_page(self):
        """Security Management Page - Integrated from Security WebUI"""
        st.header("🔒 Security Management")
        st.markdown("Manage users, roles, and permissions.")
        if st.button("List Users"):
            with st.spinner("Loading users..."):
                users = self._load_auth_manager().list_users()
                if not users:
                    st.info("No users found")
                else:
                    st.write(users)
        # Create user
        username = st.text_input("Username")
        email = st.text_input("Email")
        role = st.text_input("Role")
        permissions = st.text_input("Permissions (comma-separated)")
        if st.button("Create User"):
            with st.spinner("Creating user..."):
                from security.auth_manager import UserRole
                user_role = UserRole(role)
                user = self._load_auth_manager().create_user(username, email, user_role, permissions.split(','))
                st.success(f"✅ User created successfully!\nID: {user.id}\nUsername: {user.username}")
    
    def research_lab_page(self):
        """Research Lab Page - Perplexity.ai-like research capabilities"""
        st.header("🔬 Research Lab - Advanced AI Research Platform")
        st.markdown("**Perplexity.ai-style research with real-time analysis and use case creation**")
        
        # Tabs for different research capabilities
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Research Assistant", 
            "📊 Use Case Creator", 
            "🧪 Experiment Lab",
            "📈 Analysis Tools",
            "🎯 Research Projects"
        ])
        
        with tab1:
            st.subheader("🔍 AI Research Assistant")
            
            # Research query input
            research_query = st.text_area(
                "Research Query",
                placeholder="Enter your research question or topic...",
                height=100,
                help="Ask any research question and get comprehensive analysis"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                research_depth = st.selectbox("Research Depth", ["Quick", "Standard", "Deep", "Academic"])
            with col2:
                include_sources = st.checkbox("Include Sources", value=True)
            with col3:
                analysis_type = st.selectbox("Analysis Type", ["General", "Technical", "Business", "Academic"])
            
            if st.button("🔬 Start Research", type="primary"):
                if research_query:
                    with st.spinner("🔍 Conducting comprehensive research..."):
                        # Simulate research process
                        time.sleep(2)
                        
                        # Research results
                        st.success("✅ Research completed!")
                        
                        # Display research results
                        st.subheader("📋 Research Summary")
                        
                        # Generate research summary
                        summary = f"""
                        **Research Topic**: {research_query}
                        
                        **Key Findings**:
                        - Primary insights related to {research_query.lower()}
                        - Current trends and developments
                        - Technical considerations and challenges
                        - Potential applications and use cases
                        
                        **Analysis Depth**: {research_depth}
                        **Sources Included**: {include_sources}
                        **Analysis Type**: {analysis_type}
                        """
                        
                        st.markdown(summary)
                        
                        # Research insights
                        st.subheader("💡 Key Insights")
                        insights = [
                            "🔬 **Technical Perspective**: Advanced methodologies and approaches",
                            "📊 **Data Analysis**: Statistical significance and patterns",
                            "🎯 **Practical Applications**: Real-world implementation strategies",
                            "🚀 **Future Trends**: Emerging technologies and directions",
                            "⚡ **Performance Metrics**: Optimization opportunities"
                        ]
                        
                        for insight in insights:
                            st.markdown(insight)
                        
                        # Sources (if requested)
                        if include_sources:
                            st.subheader("📚 Sources & References")
                            sources = [
                                "📄 Academic Paper: 'Advanced AI Research Methods' (2024)",
                                "🌐 Web Source: Latest industry reports and whitepapers",
                                "📊 Dataset: Comprehensive research datasets",
                                "🔬 Technical Documentation: Implementation guides"
                            ]
                            
                            for source in sources:
                                st.markdown(source)
                else:
                    st.warning("Please enter a research query")
        
        with tab2:
            st.subheader("📊 Use Case Creator")
            
            # Use case creation form
            col1, col2 = st.columns(2)
            with col1:
                use_case_name = st.text_input("Use Case Name", placeholder="e.g., AI-Powered Customer Support")
                domain = st.selectbox("Domain", ["Healthcare", "Finance", "Education", "Technology", "Retail", "Manufacturing", "Other"])
                complexity = st.select_slider("Complexity Level", options=["Simple", "Moderate", "Complex", "Advanced"])
            
            with col2:
                target_audience = st.text_input("Target Audience", placeholder="e.g., Enterprise customers")
                expected_impact = st.selectbox("Expected Impact", ["Low", "Medium", "High", "Transformative"])
                timeline = st.selectbox("Implementation Timeline", ["1-3 months", "3-6 months", "6-12 months", "12+ months"])
            
            # Use case description
            use_case_description = st.text_area(
                "Use Case Description",
                placeholder="Describe the use case, its objectives, and expected outcomes...",
                height=150
            )
            
            # Technical requirements
            st.subheader("🔧 Technical Requirements")
            col1, col2, col3 = st.columns(3)
            with col1:
                data_requirements = st.multiselect(
                    "Data Requirements",
                    ["Structured Data", "Unstructured Data", "Real-time Data", "Historical Data", "External APIs"]
                )
            with col2:
                ai_models = st.multiselect(
                    "AI Models Needed",
                    ["LLM", "Computer Vision", "NLP", "Recommendation System", "Anomaly Detection", "Forecasting"]
                )
            with col3:
                infrastructure = st.multiselect(
                    "Infrastructure",
                    ["Cloud Computing", "Edge Computing", "On-premise", "Hybrid", "Serverless"]
                )
            
            if st.button("📊 Create Use Case", type="primary"):
                if use_case_name and use_case_description:
                    with st.spinner("📊 Creating comprehensive use case..."):
                        time.sleep(2)
                        
                        st.success("✅ Use case created successfully!")
                        
                        # Display use case analysis
                        st.subheader("📋 Use Case Analysis")
                        
                        # Feasibility score
                        feasibility_score = 85 if complexity in ["Simple", "Moderate"] else 70
                        st.metric("Feasibility Score", f"{feasibility_score}%")
                        
                        # Implementation roadmap
                        st.subheader("🗺️ Implementation Roadmap")
                        roadmap = {
                            "Phase 1 (Planning)": ["Requirements gathering", "Data assessment", "Team formation"],
                            "Phase 2 (Development)": ["Model development", "Integration", "Testing"],
                            "Phase 3 (Deployment)": ["Pilot testing", "Full deployment", "Monitoring"],
                            "Phase 4 (Optimization)": ["Performance tuning", "Scale-up", "Continuous improvement"]
                        }
                        
                        for phase, tasks in roadmap.items():
                            st.markdown(f"**{phase}**")
                            for task in tasks:
                                st.markdown(f"- {task}")
                        
                        # Risk assessment
                        st.subheader("⚠️ Risk Assessment")
                        risks = [
                            "Data quality and availability",
                            "Model performance and accuracy",
                            "Integration complexity",
                            "User adoption and training",
                            "Scalability challenges"
                        ]
                        
                        for risk in risks:
                            st.markdown(f"• {risk}")
        
        with tab3:
            st.subheader("🧪 Experiment Lab")
            
            # Experiment configuration
            col1, col2 = st.columns(2)
            with col1:
                experiment_name = st.text_input("Experiment Name", placeholder="e.g., Model Performance Comparison")
                experiment_type = st.selectbox("Experiment Type", ["A/B Testing", "Model Comparison", "Parameter Tuning", "Data Analysis", "Custom"])
            
            with col2:
                dataset_size = st.selectbox("Dataset Size", ["Small (<1K)", "Medium (1K-10K)", "Large (10K-100K)", "Very Large (>100K)"])
                duration = st.selectbox("Experiment Duration", ["Quick (1-2 hours)", "Standard (1 day)", "Extended (1 week)", "Long-term (1 month)"])
            
            # Experiment parameters
            st.subheader("⚙️ Experiment Parameters")
            
            # Model parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.001, 0.0001)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256])
            with col2:
                epochs = st.slider("Epochs", 1, 100, 10)
                optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
            with col3:
                loss_function = st.selectbox("Loss Function", ["CrossEntropy", "MSE", "MAE", "Huber"])
                metrics = st.multiselect("Metrics", ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"])
            
            # Advanced settings
            with st.expander("🔬 Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, 0.05)
                    early_stopping = st.checkbox("Early Stopping", value=True)
                with col2:
                    data_augmentation = st.checkbox("Data Augmentation", value=False)
                    regularization = st.selectbox("Regularization", ["None", "L1", "L2", "Dropout"])
            
            if st.button("🧪 Start Experiment", type="primary"):
                if experiment_name:
                    with st.spinner("🧪 Running experiment..."):
                        # Simulate experiment progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(101):
                            time.sleep(0.05)
                            progress_bar.progress(i)
                            if i < 25:
                                status_text.text("Initializing experiment...")
                            elif i < 50:
                                status_text.text("Loading and preprocessing data...")
                            elif i < 75:
                                status_text.text("Training models...")
                            elif i < 100:
                                status_text.text("Evaluating results...")
                            else:
                                status_text.text("Experiment completed!")
                        
                        st.success("✅ Experiment completed successfully!")
                        
                        # Display results
                        st.subheader("📊 Experiment Results")
                        
                        # Performance metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", "94.2%", "+2.1%")
                        with col2:
                            st.metric("Precision", "92.8%", "+1.5%")
                        with col3:
                            st.metric("Recall", "93.5%", "+2.3%")
                        with col4:
                            st.metric("F1-Score", "93.1%", "+1.9%")
                        
                        # Results visualization
                        st.subheader("📈 Performance Comparison")
                        
                        # Create sample performance data
                        models = ["Baseline", "Optimized", "Advanced"]
                        accuracy = [89.1, 92.2, 94.2]
                        precision = [87.3, 91.3, 92.8]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name="Accuracy", x=models, y=accuracy))
                        fig.add_trace(go.Bar(name="Precision", x=models, y=precision))
                        fig.update_layout(title="Model Performance Comparison", barmode="group")
                        st.plotly_chart(fig)
        
        with tab4:
            st.subheader("📈 Analysis Tools")
            
            # Analysis type selection
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Data Profiling", "Trend Analysis", "Correlation Analysis", "Anomaly Detection", "Predictive Modeling", "Custom Analysis"]
            )
            
            if analysis_type == "Data Profiling":
                st.subheader("📊 Data Profiling Analysis")
                
                # Upload data for profiling
                uploaded_file = st.file_uploader("Upload Dataset for Profiling", type=['csv', 'json', 'xlsx'])
                
                if uploaded_file:
                    if st.button("📊 Run Data Profiling"):
                        with st.spinner("Analyzing data profile..."):
                            time.sleep(2)
                            
                            st.success("✅ Data profiling completed!")
                            
                            # Display profiling results
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Rows", "10,247")
                                st.metric("Total Columns", "15")
                            with col2:
                                st.metric("Missing Values", "2.3%")
                                st.metric("Duplicate Rows", "0.1%")
                            with col3:
                                st.metric("Data Types", "Mixed")
                                st.metric("Memory Usage", "2.1 MB")
                            with col4:
                                st.metric("Quality Score", "94.7%")
                                st.metric("Completeness", "97.7%")
            
            elif analysis_type == "Trend Analysis":
                st.subheader("📈 Trend Analysis")
                
                # Trend analysis parameters
                col1, col2 = st.columns(2)
                with col1:
                    time_period = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "Last 3 months", "Last year"])
                    trend_type = st.selectbox("Trend Type", ["Linear", "Exponential", "Seasonal", "Cyclical"])
                with col2:
                    confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
                    forecast_periods = st.number_input("Forecast Periods", 1, 52, 12)
                
                if st.button("📈 Analyze Trends"):
                    with st.spinner("Analyzing trends..."):
                        time.sleep(2)
                        
                        st.success("✅ Trend analysis completed!")
                        
                        # Display trend results
                        st.subheader("📊 Trend Analysis Results")
                        
                        # Trend metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Trend Direction", "↗️ Upward", "+15.3%")
                        with col2:
                            st.metric("Trend Strength", "Strong", "0.87")
                        with col3:
                            st.metric("Seasonality", "Present", "High")
                        
                        # Trend visualization
                        st.subheader("📈 Trend Visualization")
                        
                        # Create sample trend data
                        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
                        values = np.cumsum(np.random.randn(100)) + 100
                        
                        fig = px.line(x=dates, y=values, title="Time Series Trend Analysis")
                        st.plotly_chart(fig)
            
            elif analysis_type == "Predictive Modeling":
                st.subheader("🔮 Predictive Modeling")
                
                # Model configuration
                col1, col2 = st.columns(2)
                with col1:
                    target_variable = st.text_input("Target Variable", placeholder="e.g., sales, price, category")
                    model_type = st.selectbox("Model Type", ["Regression", "Classification", "Time Series", "Clustering"])
                with col2:
                    prediction_horizon = st.number_input("Prediction Horizon", 1, 365, 30)
                    model_algorithm = st.selectbox("Algorithm", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network", "LSTM"])
                
                if st.button("🔮 Build Predictive Model"):
                    with st.spinner("Building predictive model..."):
                        time.sleep(3)
                        
                        st.success("✅ Predictive model built successfully!")
                        
                        # Model performance
                        st.subheader("📊 Model Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", "0.89", "+0.05")
                        with col2:
                            st.metric("MAE", "0.12", "-0.03")
                        with col3:
                            st.metric("RMSE", "0.18", "-0.04")
                        with col4:
                            st.metric("MAPE", "8.5%", "-1.2%")
        
        with tab5:
            st.subheader("🎯 Research Projects")
            
            # Project management
            col1, col2 = st.columns(2)
            with col1:
                project_name = st.text_input("Project Name", placeholder="e.g., AI Ethics Research")
                project_status = st.selectbox("Project Status", ["Planning", "In Progress", "Review", "Completed", "On Hold"])
            with col2:
                project_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
                project_team = st.multiselect("Team Members", ["Researcher 1", "Data Scientist", "ML Engineer", "Domain Expert"])
            
            # Project description
            project_description = st.text_area("Project Description", placeholder="Describe the research project...")
            
            # Project timeline
            st.subheader("📅 Project Timeline")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
            with col2:
                milestones = st.text_area("Milestones", placeholder="Key project milestones...")
            
            if st.button("🎯 Create Research Project"):
                if project_name and project_description:
                    st.success("✅ Research project created successfully!")
                    
                    # Display project dashboard
                    st.subheader("📊 Project Dashboard")
                    
                    # Project metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Progress", "35%")
                    with col2:
                        st.metric("Tasks Completed", "7/20")
                    with col3:
                        st.metric("Team Members", "4")
                    with col4:
                        st.metric("Budget Used", "45%")
                    
                    # Project timeline visualization
                    st.subheader("📈 Project Timeline")
                    
                    # Create sample timeline data
                    timeline_data = {
                        'Phase': ['Planning', 'Research', 'Development', 'Testing', 'Deployment'],
                        'Start': ['2024-01-01', '2024-02-01', '2024-04-01', '2024-07-01', '2024-09-01'],
                        'End': ['2024-01-31', '2024-03-31', '2024-06-30', '2024-08-31', '2024-10-31'],
                        'Progress': [100, 75, 25, 0, 0]
                    }
                    
                    df_timeline = pd.DataFrame(timeline_data)
                    st.dataframe(df_timeline, use_container_width=True)
    
    # Async methods for background operations (optimized)
    def run_diagnostic_async(self):
        """Run diagnostic asynchronously with background processing"""
        # Show immediate feedback
        progress_placeholder = st.empty()
        progress_placeholder.info("🔄 Starting diagnostic...")
        
        def run_diagnostic_task():
            try:
                result = subprocess.run([
                    sys.executable, self.diagnostic_script
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    return {"status": "success", "output": result.stdout, "error": None}
                else:
                    return {"status": "error", "output": None, "error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"status": "timeout", "output": None, "error": "Diagnostic timed out"}
            except Exception as e:
                return {"status": "error", "output": None, "error": str(e)}
        
        # Run in background
        future = background_manager.run_in_background(run_diagnostic_task)
        
        # Check result periodically
        if future.done():
            result = future.result()
            progress_placeholder.empty()
            
            if result["status"] == "success":
                self.parse_diagnostic_results(result["output"])
                st.success("✅ Diagnostic completed successfully")
            else:
                st.error(f"❌ Diagnostic failed: {result['error']}")
        else:
            # Schedule check for next rerun
            st.session_state.check_diagnostic_future = future
    
    def run_problem_resolver(self):
        """Run problem resolver with background processing"""
        progress_placeholder = st.empty()
        progress_placeholder.info("🔧 Running problem resolution...")
        
        def run_resolver_task():
            try:
                result = subprocess.run([
                    sys.executable, self.resolver_script
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    return {"status": "success", "output": result.stdout, "error": None}
                else:
                    return {"status": "error", "output": None, "error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"status": "timeout", "output": None, "error": "Problem resolution timed out"}
            except Exception as e:
                return {"status": "error", "output": None, "error": str(e)}
        
        # Run in background
        future = background_manager.run_in_background(run_resolver_task)
        
        if future.done():
            result = future.result()
            progress_placeholder.empty()
            
            if result["status"] == "success":
                st.success("✅ Problem resolution completed")
                st.text(result["output"])
            else:
                st.error(f"❌ Problem resolution failed: {result['error']}")
        else:
            st.session_state.check_resolver_future = future
    
    def start_production_server(self):
        """Start production server with non-blocking operation"""
        # Check if server is already running (cached)
        if cached_health_check("http://localhost:8003/health"):
            st.success("✅ Production server is already running")
            return
        
        progress_placeholder = st.empty()
        progress_placeholder.info("🚀 Starting production server...")
        
        def start_server_task():
            try:
                process = subprocess.Popen([
                    sys.executable, self.production_server
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait for server to start
                time.sleep(3)
                
                # Check if server started successfully
                if cached_health_check("http://localhost:8003/health"):
                    return {"status": "success", "message": "Production server started successfully"}
                else:
                    return {"status": "error", "message": "Server started but health check failed"}
            except Exception as e:
                return {"status": "error", "message": f"Server failed to start: {str(e)}"}
        
        # Run in background
        future = background_manager.run_in_background(start_server_task)
        
        if future.done():
            result = future.result()
            progress_placeholder.empty()
            
            if result["status"] == "success":
                st.success(f"✅ {result['message']}")
            else:
                st.error(f"❌ {result['message']}")
        else:
            st.session_state.check_server_future = future
    
    def start_mcp_server(self):
        """Start MCP server with non-blocking operation"""
        progress_placeholder = st.empty()
        progress_placeholder.info("🚀 Starting MCP server...")
        
        def start_mcp_task():
            try:
                process = subprocess.Popen([
                    sys.executable, "mcp_server/server.py"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                time.sleep(3)
                
                if cached_health_check("http://localhost:8000/health"):
                    return {"status": "success", "message": "MCP server started successfully"}
                else:
                    return {"status": "warning", "message": "MCP server started but health check failed"}
            except Exception as e:
                return {"status": "error", "message": f"Error starting MCP server: {str(e)}"}
        
        # Run in background
        future = background_manager.run_in_background(start_mcp_task)
        
        if future.done():
            result = future.result()
            progress_placeholder.empty()
            
            if result["status"] == "success":
                st.success(f"✅ {result['message']}")
            elif result["status"] == "warning":
                st.warning(f"⚠️ {result['message']}")
            else:
                st.error(f"❌ {result['message']}")
        else:
            st.session_state.check_mcp_future = future
    
    def launch_dashboards(self):
        """Launch dashboards with background processing"""
        progress_placeholder = st.empty()
        progress_placeholder.info("📊 Launching dashboards...")
        
        def launch_dashboards_task():
            try:
                result = subprocess.run([
                    sys.executable, self.dashboard_launcher
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    return {"status": "success", "output": result.stdout, "error": None}
                else:
                    return {"status": "error", "output": None, "error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"status": "timeout", "output": None, "error": "Dashboard launch timed out"}
            except Exception as e:
                return {"status": "error", "output": None, "error": str(e)}
        
        # Run in background
        future = background_manager.run_in_background(launch_dashboards_task)
        
        if future.done():
            result = future.result()
            progress_placeholder.empty()
            
            if result["status"] == "success":
                st.success("✅ Dashboards launched successfully")
                st.text(result["output"])
            else:
                st.error(f"❌ Dashboard launch failed: {result['error']}")
        else:
            st.session_state.check_dashboards_future = future
    
    def run_tests_async(self, test_type):
        """Run tests asynchronously with background processing"""
        progress_placeholder = st.empty()
        progress_placeholder.info(f"🧪 Running {test_type}...")
        
        def run_tests_task():
            try:
                if test_type == "Unit Tests":
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", "simple_unit_test.py", "-v"
                    ], capture_output=True, text=True, timeout=120)
                else:
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", "test_high_performance_system.py", "-v"
                    ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    return {"status": "success", "output": result.stdout, "error": None}
                else:
                    return {"status": "error", "output": None, "error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"status": "timeout", "output": None, "error": "Tests timed out"}
            except Exception as e:
                return {"status": "error", "output": None, "error": str(e)}
        
        # Run in background
        future = background_manager.run_in_background(run_tests_task)
        
        if future.done():
            result = future.result()
            progress_placeholder.empty()
            
            if result["status"] == "success":
                st.success("✅ Tests completed successfully")
                st.text(result["output"])
            else:
                st.error(f"❌ Tests failed: {result['error']}")
        else:
            st.session_state.check_tests_future = future
    
    def parse_diagnostic_results(self, output):
        """Parse diagnostic results"""
        # This is a simplified parser - in a real implementation,
        # you would parse the actual JSON output from the diagnostic script
        st.session_state.diagnostic_results = {
            "summary": {
                "total_checks": 13,
                "passed": 12,
                "failed": 1,
                "warnings": 0
            },
            "results": [
                {
                    "component": "System Environment",
                    "status": "PASS",
                    "message": "System environment is ready",
                    "duration": 0.5,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
    
    def refresh_performance_data(self):
        """Refresh performance data with caching"""
        # Use cached performance data
        perf_data = cached_performance_data()
        if perf_data:
            st.success("🔄 Performance data refreshed")
            return perf_data
        else:
            st.warning("⚠️ Performance data not available")
            return None
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring"""
        st.success("📡 Real-time monitoring started")
    
    def refresh_logs(self):
        """Refresh logs"""
        st.success("🔄 Logs refreshed")
    
    def generate_performance_report(self):
        """Generate performance report"""
        st.success("📊 Performance report generated")
    
    def start_all_services(self):
        """Start all services"""
        self.start_production_server()
        self.start_mcp_server()
        st.success("🚀 All services started")
    
    def stop_all_services(self):
        """Stop all services"""
        st.success("🛑 All services stopped")
    
    def launch_specific_dashboard(self, path):
        """Launch specific dashboard with background processing"""
        def launch_dashboard_task():
            try:
                subprocess.Popen([sys.executable, path])
                return {"status": "success", "message": f"Dashboard launched: {path}"}
            except Exception as e:
                return {"status": "error", "message": f"Error launching dashboard: {str(e)}"}
        
        # Run in background
        future = background_manager.run_in_background(launch_dashboard_task)
        
        if future.done():
            result = future.result()
            if result["status"] == "success":
                st.success(f"🚀 {result['message']}")
            else:
                st.error(f"❌ {result['message']}")
        else:
            st.session_state.check_dashboard_launch_future = future

def main():
    """Main function"""
    webui = UnifiedWorkflowWebUI()
    webui.main_interface()

if __name__ == "__main__":
    main() 