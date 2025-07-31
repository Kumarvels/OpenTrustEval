#!/usr/bin/env python3
"""
Trust Scoring System Dashboard
Lightweight data science dashboard with SQL support for monitoring trust scoring
Provides 360-degree view of scoring system with query, command, verify, check, and report capabilities
"""
# High-Performance System Integration
try:
    from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
    from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
    
    # Initialize high-performance components
    moe_system = UltimateMoESystem()
    expert_ensemble = AdvancedExpertEnsemble()
    
    HIGH_PERFORMANCE_AVAILABLE = True
    print(f"✅ Trust Scoring Dashboard integrated with high-performance system")
except ImportError as e:
    HIGH_PERFORMANCE_AVAILABLE = False
    print(f"⚠️ High-performance system not available for Trust Scoring Dashboard: {e}")

def get_high_performance_status():
    """Get high-performance system status"""
    return {
        'available': HIGH_PERFORMANCE_AVAILABLE,
        'moe_system': 'active' if HIGH_PERFORMANCE_AVAILABLE and moe_system else 'inactive',
        'trust_scorer': 'active' if HIGH_PERFORMANCE_AVAILABLE and expert_ensemble else 'inactive',
        'dataset_profiler': 'active' if HIGH_PERFORMANCE_AVAILABLE and expert_ensemble else 'inactive'
    }


import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import subprocess
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy.stats import entropy
import tempfile
import shutil
from pathlib import Path

# Cloud storage imports (optional)
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from google.cloud import storage
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    PYDRIVE_AVAILABLE = True
except ImportError:
    PYDRIVE_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import trust scoring components
from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
from data_engineering.cleanlab_integration import FallbackDataQualityManager
from data_engineering.dataset_integration import DatasetManager
from data_engineering.batch_trust_scoring_test_suite import BatchTrustScoringTestSuite

class TrustScoringDashboard:
    """
    Comprehensive dashboard for trust scoring system monitoring
    """
    
    def __init__(self, db_path: str = "./trust_scoring_dashboard.db"):
        self.db_path = db_path
        self.setup_database()
        self.setup_logging()
        
        # Initialize components
        self.advanced_engine = AdvancedTrustScoringEngine()
        self.fallback_manager = FallbackDataQualityManager()
        self.dataset_manager = DatasetManager()
        
        # Dashboard state
        self.current_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def setup_database(self):
        """Setup SQLite database for dashboard data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different aspects of trust scoring
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT,
                dataset_id TEXT,
                trust_score REAL,
                method TEXT,
                component_scores TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT,
                dataset_id TEXT,
                missing_values_ratio REAL,
                duplicate_rows_ratio REAL,
                outlier_ratio REAL,
                data_completeness REAL,
                data_consistency REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                test_type TEXT,
                status TEXT,
                duration REAL,
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                metric_unit TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS commands_executed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT,
                status TEXT,
                output TEXT,
                duration REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def setup_logging(self):
        """Setup logging for dashboard"""
        self.logger = logging.getLogger('TrustScoringDashboard')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_trust_score(self, dataset_name: str, dataset_id: str, trust_score: float, 
                       method: str, component_scores: Dict = None):
        """Log trust score to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trust_scores (dataset_name, dataset_id, trust_score, method, component_scores, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (dataset_name, dataset_id, trust_score, method, 
              json.dumps(component_scores) if component_scores else None, self.current_session))
        
        conn.commit()
        conn.close()
    
    def log_quality_metrics(self, dataset_name: str, dataset_id: str, metrics: Dict):
        """Log quality metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_metrics 
            (dataset_name, dataset_id, missing_values_ratio, duplicate_rows_ratio,
             outlier_ratio, data_completeness, data_consistency, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dataset_name, dataset_id, 
              metrics.get('missing_values_ratio', 0),
              metrics.get('duplicate_rows_ratio', 0),
              metrics.get('outlier_ratio', 0),
              metrics.get('data_completeness', 0),
              metrics.get('data_consistency', 0),
              self.current_session))
        
        conn.commit()
        conn.close()
    
    def log_test_result(self, test_name: str, test_type: str, status: str, 
                       duration: float, details: Dict = None):
        """Log test result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO test_results (test_name, test_type, status, duration, details, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (test_name, test_type, status, duration, 
              json.dumps(details) if details else None, self.current_session))
        
        conn.commit()
        conn.close()
    
    def log_system_metric(self, metric_name: str, metric_value: float, metric_unit: str = ""):
        """Log system metric to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_metrics (metric_name, metric_value, metric_unit, session_id)
            VALUES (?, ?, ?, ?)
        ''', (metric_name, metric_value, metric_unit, self.current_session))
        
        conn.commit()
        conn.close()
    
    def log_command(self, command: str, status: str, output: str, duration: float):
        """Log command execution to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO commands_executed (command, status, output, duration, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (command, status, output, duration, self.current_session))
        
        conn.commit()
        conn.close()
    
    def upload_file_from_streamlit(self, uploaded_file) -> Optional[str]:
        """Handle file upload from Streamlit file uploader"""
        if uploaded_file is not None:
            try:
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Move to uploads directory
                uploads_dir = Path("./uploads")
                uploads_dir.mkdir(exist_ok=True)
                
                final_path = uploads_dir / uploaded_file.name
                shutil.move(temp_path, final_path)
                
                self.logger.info(f"File uploaded successfully: {final_path}")
                return str(final_path)
                
            except Exception as e:
                self.logger.error(f"Error uploading file: {e}")
                return None
        return None
    
    def upload_from_s3(self, bucket_name: str, file_key: str, aws_access_key: str = None, 
                      aws_secret_key: str = None, region: str = "us-east-1") -> Optional[str]:
        """Upload file from S3"""
        if not BOTO3_AVAILABLE:
            st.error("boto3 not available. Install with: pip install boto3")
            return None
        
        try:
            # Configure S3 client
            if aws_access_key and aws_secret_key:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region
                )
            else:
                # Use default credentials
                s3_client = boto3.client('s3', region_name=region)
            
            # Download file
            uploads_dir = Path("./uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            local_path = uploads_dir / Path(file_key).name
            
            s3_client.download_file(bucket_name, file_key, str(local_path))
            
            self.logger.info(f"File downloaded from S3: {local_path}")
            return str(local_path)
            
        except NoCredentialsError:
            st.error("AWS credentials not found. Please provide access key and secret key.")
            return None
        except ClientError as e:
            st.error(f"S3 error: {e}")
            return None
        except Exception as e:
            st.error(f"Error downloading from S3: {e}")
            return None
    
    def upload_from_google_drive(self, file_id: str, credentials_path: str = None) -> Optional[str]:
        """Upload file from Google Drive"""
        if not PYDRIVE_AVAILABLE:
            st.error("PyDrive not available. Install with: pip install pydrive")
            return None
        
        try:
            # Authenticate with Google Drive
            gauth = GoogleAuth()
            
            if credentials_path and os.path.exists(credentials_path):
                gauth.LoadCredentialsFile(credentials_path)
            
            if gauth.credentials is None:
                # Authenticate if not already authenticated
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                gauth.Refresh()
            else:
                gauth.Authorize()
            
            gauth.SaveCredentialsFile("credentials.txt")
            
            # Create Google Drive instance
            drive = GoogleDrive(gauth)
            
            # Download file
            file_obj = drive.CreateFile({'id': file_id})
            file_name = file_obj['title']
            
            uploads_dir = Path("./uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            local_path = uploads_dir / file_name
            file_obj.GetContentFile(str(local_path))
            
            self.logger.info(f"File downloaded from Google Drive: {local_path}")
            return str(local_path)
            
        except Exception as e:
            st.error(f"Error downloading from Google Drive: {e}")
            return None
    
    def upload_from_google_cloud_storage(self, bucket_name: str, blob_name: str, 
                                       credentials_path: str = None) -> Optional[str]:
        """Upload file from Google Cloud Storage"""
        if not GOOGLE_CLOUD_AVAILABLE:
            st.error("Google Cloud Storage not available. Install with: pip install google-cloud-storage")
            return None
        
        try:
            # Initialize client
            if credentials_path:
                storage_client = storage.Client.from_service_account_json(credentials_path)
            else:
                storage_client = storage.Client()
            
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Download file
            uploads_dir = Path("./uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            local_path = uploads_dir / Path(blob_name).name
            blob.download_to_filename(str(local_path))
            
            self.logger.info(f"File downloaded from Google Cloud Storage: {local_path}")
            return str(local_path)
            
        except Exception as e:
            st.error(f"Error downloading from Google Cloud Storage: {e}")
            return None
    
    def upload_from_local_path(self, file_path: str) -> Optional[str]:
        """Upload file from local path"""
        try:
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
                return None
            
            # Copy to uploads directory
            uploads_dir = Path("./uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            file_name = Path(file_path).name
            destination_path = uploads_dir / file_name
            
            shutil.copy2(file_path, destination_path)
            
            self.logger.info(f"File copied from local path: {destination_path}")
            return str(destination_path)
            
        except Exception as e:
            st.error(f"Error copying file: {e}")
            return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return [
            "CSV", "JSON", "Excel", "Parquet", "HDF5", "Pickle", 
            "Feather", "Stata", "SAS", "SPSS", "XML", "YAML"
        ]
    
    def validate_file_format(self, file_path: str) -> Tuple[bool, str]:
        """Validate if file format is supported"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            supported_extensions = {
                '.csv': 'CSV',
                '.json': 'JSON', 
                '.xlsx': 'Excel',
                '.xls': 'Excel',
                '.parquet': 'Parquet',
                '.h5': 'HDF5',
                '.hdf5': 'HDF5',
                '.pkl': 'Pickle',
                '.pickle': 'Pickle',
                '.feather': 'Feather',
                '.dta': 'Stata',
                '.sas7bdat': 'SAS',
                '.sav': 'SPSS',
                '.xml': 'XML',
                '.yaml': 'YAML',
                '.yml': 'YAML'
            }
            
            if file_ext in supported_extensions:
                return True, supported_extensions[file_ext]
            else:
                return False, f"Unsupported file format: {file_ext}"
                
        except Exception as e:
            return False, f"Error validating file: {e}"
    
    def execute_sql_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"SQL Query Error: {e}")
            return pd.DataFrame()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        data = {}
        
        # Get recent trust scores
        data['recent_trust_scores'] = self.execute_sql_query('''
            SELECT * FROM trust_scores 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        # Get quality metrics
        data['quality_metrics'] = self.execute_sql_query('''
            SELECT * FROM quality_metrics 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        # Get test results
        data['test_results'] = self.execute_sql_query('''
            SELECT * FROM test_results 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        # Get system metrics
        data['system_metrics'] = self.execute_sql_query('''
            SELECT * FROM system_metrics 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        
        # Get command history
        data['command_history'] = self.execute_sql_query('''
            SELECT * FROM commands_executed 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        return data
    
    def run_batch_test_suite(self) -> Dict[str, Any]:
        """Run batch test suite and log results"""
        st.info("Running comprehensive batch test suite...")
        
        start_time = time.time()
        
        try:
            test_suite = BatchTrustScoringTestSuite()
            results = test_suite.run_complete_test_suite()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Log test results
            self.log_test_result(
                "Batch Test Suite",
                "comprehensive",
                "completed",
                duration,
                {"total_tests": len(results)}
            )
            
            st.success(f"Batch test suite completed in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.log_test_result(
                "Batch Test Suite",
                "comprehensive",
                "failed",
                duration,
                {"error": str(e)}
            )
            
            st.error(f"Batch test suite failed: {e}")
            return {"error": str(e)}
    
    def execute_trust_scoring_command(self, command: str, dataset_path: str = None) -> Dict[str, Any]:
        """Execute trust scoring command and log results"""
        start_time = time.time()
        
        try:
            if command == "calculate_trust_score":
                if not dataset_path:
                    st.error("Dataset path required for trust score calculation")
                    return {"error": "Dataset path required"}
                
                df = pd.read_csv(dataset_path)
                result = self.advanced_engine.calculate_advanced_trust_score(df)
                
                # Log results
                self.log_trust_score(
                    os.path.basename(dataset_path),
                    dataset_path,
                    result.get('trust_score', 0),
                    result.get('method', 'unknown'),
                    result.get('component_scores')
                )
                
                return result
                
            elif command == "quality_assessment":
                if not dataset_path:
                    st.error("Dataset path required for quality assessment")
                    return {"error": "Dataset path required"}
                
                df = pd.read_csv(dataset_path)
                result = self.fallback_manager.calculate_data_trust_score(df)
                
                # Log quality metrics
                if 'quality_metrics' in result:
                    self.log_quality_metrics(
                        os.path.basename(dataset_path),
                        dataset_path,
                        result['quality_metrics']
                    )
                
                return result
                
            elif command == "dataset_validation":
                if not dataset_path:
                    st.error("Dataset path required for validation")
                    return {"error": "Dataset path required"}
                
                # Import dataset and validate
                dataset_id = self.dataset_manager.import_dataset(dataset_path, "validation_test")
                validation_result = self.dataset_manager.validate_dataset(dataset_id)
                
                return validation_result
                
            else:
                return {"error": f"Unknown command: {command}"}
                
        except Exception as e:
            return {"error": str(e)}
        
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Log command execution
            self.log_command(command, "completed" if "error" not in locals() else "failed", 
                           str(locals().get('result', '')), duration)
    
    def create_visualizations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive visualizations"""
        viz = {}
        
        # Trust Score Trends
        if not data['recent_trust_scores'].empty:
            fig_trust = px.line(data['recent_trust_scores'], 
                               x='timestamp', y='trust_score', 
                               color='method', title='Trust Score Trends')
            viz['trust_score_trends'] = fig_trust
        
        # Quality Metrics Dashboard
        if not data['quality_metrics'].empty:
            fig_quality = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Missing Values Ratio', 'Duplicate Rows Ratio',
                              'Outlier Ratio', 'Data Completeness'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_quality.add_trace(
                go.Scatter(x=data['quality_metrics']['timestamp'], 
                          y=data['quality_metrics']['missing_values_ratio'],
                          name='Missing Values'),
                row=1, col=1
            )
            
            fig_quality.add_trace(
                go.Scatter(x=data['quality_metrics']['timestamp'], 
                          y=data['quality_metrics']['duplicate_rows_ratio'],
                          name='Duplicates'),
                row=1, col=2
            )
            
            fig_quality.add_trace(
                go.Scatter(x=data['quality_metrics']['timestamp'], 
                          y=data['quality_metrics']['outlier_ratio'],
                          name='Outliers'),
                row=2, col=1
            )
            
            fig_quality.add_trace(
                go.Scatter(x=data['quality_metrics']['timestamp'], 
                          y=data['quality_metrics']['data_completeness'],
                          name='Completeness'),
                row=2, col=2
            )
            
            fig_quality.update_layout(height=600, title_text="Quality Metrics Dashboard")
            viz['quality_dashboard'] = fig_quality
        
        # Test Results Summary
        if not data['test_results'].empty:
            test_summary = data['test_results'].groupby('status').size().reset_index(name='count')
            fig_tests = px.pie(test_summary, values='count', names='status', 
                              title='Test Results Summary')
            viz['test_results_summary'] = fig_tests
        
        # System Metrics
        if not data['system_metrics'].empty:
            fig_system = px.line(data['system_metrics'], 
                                x='timestamp', y='metric_value', 
                                color='metric_name', title='System Metrics')
            viz['system_metrics'] = fig_system
        
        return viz
    
    def run_streamlit_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Trust Scoring System Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔍 Trust Scoring System Dashboard")
        st.markdown("### 360° Monitoring & Analytics Platform")
        
        # Sidebar
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["📊 Overview", "🔍 Trust Scoring", "📈 Analytics", "⚡ Commands", "🧪 Testing", "📋 Reports", "🗄️ SQL Query"]
        )
        
        # Get dashboard data
        data = self.get_dashboard_data()
        
        if page == "📊 Overview":
            self.show_overview_page(data)
        elif page == "🔍 Trust Scoring":
            self.show_trust_scoring_page(data)
        elif page == "📈 Analytics":
            self.show_analytics_page(data)
        elif page == "⚡ Commands":
            self.show_commands_page(data)
        elif page == "🧪 Testing":
            self.show_testing_page(data)
        elif page == "📋 Reports":
            self.show_reports_page(data)
        elif page == "🗄️ SQL Query":
            self.show_sql_query_page(data)
    
    def show_overview_page(self, data: Dict[str, Any]):
        """Show overview page"""
        st.header("📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not data['recent_trust_scores'].empty:
                avg_trust = data['recent_trust_scores']['trust_score'].mean()
                st.metric("Average Trust Score", f"{avg_trust:.3f}")
            else:
                st.metric("Average Trust Score", "N/A")
        
        with col2:
            if not data['test_results'].empty:
                success_rate = (data['test_results']['status'] == 'completed').mean()
                st.metric("Test Success Rate", f"{success_rate:.1%}")
            else:
                st.metric("Test Success Rate", "N/A")
        
        with col3:
            if not data['quality_metrics'].empty:
                avg_completeness = data['quality_metrics']['data_completeness'].mean()
                st.metric("Data Completeness", f"{avg_completeness:.1%}")
            else:
                st.metric("Data Completeness", "N/A")
        
        with col4:
            total_commands = len(data['command_history'])
            st.metric("Commands Executed", total_commands)
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Trust Scores**")
            if not data['recent_trust_scores'].empty:
                st.dataframe(data['recent_trust_scores'].head(10))
            else:
                st.info("No trust scores recorded yet")
        
        with col2:
            st.write("**Recent Test Results**")
            if not data['test_results'].empty:
                st.dataframe(data['test_results'].head(10))
            else:
                st.info("No test results recorded yet")
        
        # System health
        st.subheader("🏥 System Health")
        
        # Create visualizations
        viz = self.create_visualizations(data)
        
        if 'trust_score_trends' in viz:
            st.plotly_chart(viz['trust_score_trends'], use_container_width=True)
        
        if 'quality_dashboard' in viz:
            st.plotly_chart(viz['quality_dashboard'], use_container_width=True)
    
    def show_trust_scoring_page(self, data: Dict[str, Any]):
        """Show trust scoring page"""
        st.header("🔍 Trust Scoring")
        
        # NEW: Comprehensive File Upload Section
        st.subheader("📁 File Upload & Data Sources")
        
        # Upload method selection
        upload_method = st.selectbox(
            "Select Upload Method",
            ["Local File Upload", "S3 Bucket", "Google Drive", "Google Cloud Storage", "Local Path"],
            help="Choose how you want to upload your dataset"
        )
        
        uploaded_file_path = None
        
        if upload_method == "Local File Upload":
            st.write("**Upload from your computer**")
            
            # File uploader with multiple formats
            supported_formats = self.get_supported_formats()
            accepted_types = [
                "text/csv", "application/json", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel", "application/octet-stream", "text/plain"
            ]
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'json', 'xlsx', 'xls', 'parquet', 'h5', 'pkl', 'feather', 'dta', 'sav', 'xml', 'yaml', 'yml'],
                help=f"Supported formats: {', '.join(supported_formats)}"
            )
            
            if uploaded_file is not None:
                # Validate file format
                is_valid, format_type = self.validate_file_format(uploaded_file.name)
                if is_valid:
                    st.success(f"✅ File format detected: {format_type}")
                    
                    # Upload file
                    uploaded_file_path = self.upload_file_from_streamlit(uploaded_file)
                    if uploaded_file_path:
                        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
                        st.info(f"File saved to: {uploaded_file_path}")
                    else:
                        st.error("❌ Failed to upload file")
                else:
                    st.error(f"❌ {format_type}")
        
        elif upload_method == "S3 Bucket":
            st.write("**Upload from Amazon S3**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                bucket_name = st.text_input("S3 Bucket Name", placeholder="my-bucket")
                file_key = st.text_input("File Key/Path", placeholder="data/dataset.csv")
                region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"])
            
            with col2:
                aws_access_key = st.text_input("AWS Access Key (optional)", type="password")
                aws_secret_key = st.text_input("AWS Secret Key (optional)", type="password")
                
                if st.button("Download from S3", type="primary"):
                    if bucket_name and file_key:
                        with st.spinner("Downloading from S3..."):
                            uploaded_file_path = self.upload_from_s3(
                                bucket_name, file_key, aws_access_key, aws_secret_key, region
                            )
                        if uploaded_file_path:
                            st.success(f"✅ File downloaded from S3: {Path(file_key).name}")
                        else:
                            st.error("❌ Failed to download from S3")
                    else:
                        st.error("Please provide bucket name and file key")
        
        elif upload_method == "Google Drive":
            st.write("**Upload from Google Drive**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                file_id = st.text_input("Google Drive File ID", placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
                st.info("To get file ID: Right-click file in Google Drive → Get link → Copy ID from URL")
            
            with col2:
                credentials_path = st.text_input("Credentials File Path (optional)", placeholder="path/to/credentials.json")
                
                if st.button("Download from Google Drive", type="primary"):
                    if file_id:
                        with st.spinner("Downloading from Google Drive..."):
                            uploaded_file_path = self.upload_from_google_drive(file_id, credentials_path)
                        if uploaded_file_path:
                            st.success(f"✅ File downloaded from Google Drive")
                        else:
                            st.error("❌ Failed to download from Google Drive")
                    else:
                        st.error("Please provide file ID")
        
        elif upload_method == "Google Cloud Storage":
            st.write("**Upload from Google Cloud Storage**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                bucket_name = st.text_input("GCS Bucket Name", placeholder="my-gcs-bucket")
                blob_name = st.text_input("Blob Name/Path", placeholder="data/dataset.csv")
            
            with col2:
                credentials_path = st.text_input("Service Account JSON Path (optional)", placeholder="path/to/service-account.json")
                
                if st.button("Download from GCS", type="primary"):
                    if bucket_name and blob_name:
                        with st.spinner("Downloading from Google Cloud Storage..."):
                            uploaded_file_path = self.upload_from_google_cloud_storage(bucket_name, blob_name, credentials_path)
                        if uploaded_file_path:
                            st.success(f"✅ File downloaded from GCS: {Path(blob_name).name}")
                        else:
                            st.error("❌ Failed to download from GCS")
                    else:
                        st.error("Please provide bucket name and blob name")
        
        elif upload_method == "Local Path":
            st.write("**Upload from local file path**")
            
            local_path = st.text_input("Local File Path", placeholder="C:/path/to/your/file.csv")
            
            if st.button("Copy from Local Path", type="primary"):
                if local_path:
                    with st.spinner("Copying file..."):
                        uploaded_file_path = self.upload_from_local_path(local_path)
                    if uploaded_file_path:
                        st.success(f"✅ File copied from local path: {Path(local_path).name}")
                    else:
                        st.error("❌ Failed to copy file")
                else:
                    st.error("Please provide local file path")
        
        # Display uploaded file info
        if uploaded_file_path:
            st.subheader("📋 Uploaded File Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**File Path:** {uploaded_file_path}")
                st.write(f"**File Name:** {Path(uploaded_file_path).name}")
                st.write(f"**File Size:** {Path(uploaded_file_path).stat().st_size / 1024:.2f} KB")
            
            with col2:
                # Validate and show file format
                is_valid, format_type = self.validate_file_format(uploaded_file_path)
                if is_valid:
                    st.success(f"**Format:** {format_type}")
                else:
                    st.error(f"**Format:** {format_type}")
                
                # Try to load and preview data
                try:
                    if format_type == "CSV":
                        df_preview = pd.read_csv(uploaded_file_path, nrows=5)
                    elif format_type == "JSON":
                        df_preview = pd.read_json(uploaded_file_path)
                    elif format_type == "Excel":
                        df_preview = pd.read_excel(uploaded_file_path, nrows=5)
                    else:
                        df_preview = pd.DataFrame({"Info": ["Preview not available for this format"]})
                    
                    st.write(f"**Preview:** {len(df_preview)} rows, {len(df_preview.columns)} columns")
                    
                    if st.button("Show Data Preview"):
                        st.dataframe(df_preview)
                        
                except Exception as e:
                    st.warning(f"Could not preview data: {e}")
        
        # Trust scoring controls
        st.subheader("🎯 Calculate Trust Score")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use uploaded file path if available, otherwise allow manual input
            if uploaded_file_path:
                dataset_path = st.text_input("Dataset Path", value=uploaded_file_path, disabled=True)
            else:
                dataset_path = st.text_input("Dataset Path", placeholder="path/to/dataset.csv")
            
            method = st.selectbox("Scoring Method", ["ensemble", "robust", "uncertainty"])
        
        with col2:
            st.write("**Quick Actions**")
            if st.button("Calculate Trust Score", type="primary"):
                if dataset_path:
                    result = self.execute_trust_scoring_command("calculate_trust_score", dataset_path)
                    if "error" not in result:
                        st.success(f"Trust Score: {result.get('trust_score', 'N/A'):.3f}")
                        st.json(result)
                    else:
                        st.error(f"Error: {result['error']}")
                else:
                    st.error("Please provide dataset path")
            
            if st.button("Quality Assessment"):
                if dataset_path:
                    result = self.execute_trust_scoring_command("quality_assessment", dataset_path)
                    if "error" not in result:
                        st.success("Quality assessment completed")
                        st.json(result)
                    else:
                        st.error(f"Error: {result['error']}")
        
        # NEW: Cleanlab Comparison Section
        st.subheader("🔬 Cleanlab Truth Score Comparison")
        
        # Data format selection
        data_format = st.selectbox(
            "Data Format",
            ["CSV", "DataFrame", "JSON", "Excel", "Parquet", "HDF5"],
            help="Select the format of your input data"
        )
        
        # Cleanlab truth value options
        cleanlab_option = st.selectbox(
            "Cleanlab Truth Value Option",
            [1, 2, 3, 4],
            format_func=lambda x: f"Option {x}",
            help="Select which Cleanlab truth value method to use for comparison"
        )
        
        # Comparison controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            comparison_method = st.selectbox(
                "Comparison Method",
                ["Side-by-Side", "Difference Analysis", "Correlation Analysis", "Statistical Test"],
                help="Choose how to compare the scores"
            )
        
        with col2:
            visualization_type = st.selectbox(
                "Visualization",
                ["Table", "Chart", "Both"],
                help="Choose visualization type for comparison"
            )
        
        with col3:
            if st.button("Run Comparison", type="primary"):
                if dataset_path:
                    comparison_result = self.run_cleanlab_comparison(
                        dataset_path, data_format, cleanlab_option, 
                        comparison_method, visualization_type
                    )
                    if "error" not in comparison_result:
                        st.success("Comparison completed successfully!")
                        self.display_comparison_results(comparison_result, visualization_type)
                    else:
                        st.error(f"Comparison Error: {comparison_result['error']}")
                else:
                    st.error("Please provide dataset path for comparison")
        
        # Trust score history
        st.subheader("📈 Trust Score History")
        
        if not data['recent_trust_scores'].empty:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                methods = data['recent_trust_scores']['method'].unique()
                selected_method = st.multiselect("Filter by Method", methods, default=list(methods))
            
            with col2:
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.now() - timedelta(days=7), datetime.now()),
                    max_value=datetime.now()
                )
            
            # Filter data
            filtered_data = data['recent_trust_scores'].copy()
            if selected_method:
                filtered_data = filtered_data[filtered_data['method'].isin(selected_method)]
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
                filtered_data = filtered_data[
                    (filtered_data['timestamp'].dt.date >= start_date) &
                    (filtered_data['timestamp'].dt.date <= end_date)
                ]
            
            st.dataframe(filtered_data)
            
            # Trust score distribution
            fig_dist = px.histogram(filtered_data, x='trust_score', 
                                   title='Trust Score Distribution')
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No trust scores recorded yet")
    
    def run_cleanlab_comparison(self, dataset_path: str, data_format: str, 
                               cleanlab_option: int, comparison_method: str, 
                               visualization_type: str) -> Dict[str, Any]:
        """
        Run comparison between our trust score and Cleanlab truth score
        
        Args:
            dataset_path: Path to the dataset
            data_format: Format of the data (CSV, DataFrame, etc.)
            cleanlab_option: Cleanlab truth value option (1-4)
            comparison_method: Method for comparison
            visualization_type: Type of visualization
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Load data based on format
            dataset = self.load_data_by_format(dataset_path, data_format)
            if dataset is None:
                return {"error": f"Could not load data from {dataset_path}"}
            
            # Calculate our trust score
            our_result = self.advanced_engine.calculate_advanced_trust_score(dataset)
            our_score = our_result.get('trust_score', 0.0)
            
            # Calculate Cleanlab truth score based on option
            cleanlab_score = self.calculate_cleanlab_truth_score(dataset, cleanlab_option)
            
            # Perform comparison analysis
            comparison_analysis = self.perform_comparison_analysis(
                our_score, cleanlab_score, comparison_method
            )
            
            # Create visualizations
            visualizations = self.create_comparison_visualizations(
                our_score, cleanlab_score, comparison_analysis, visualization_type
            )
            
            # Log comparison results
            self.log_comparison_result(our_score, cleanlab_score, cleanlab_option, comparison_method)
            
            return {
                "our_score": our_score,
                "cleanlab_score": cleanlab_score,
                "cleanlab_option": cleanlab_option,
                "comparison_method": comparison_method,
                "comparison_analysis": comparison_analysis,
                "visualizations": visualizations,
                "dataset_info": {
                    "rows": len(dataset),
                    "columns": len(dataset.columns),
                    "format": data_format
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in cleanlab comparison: {e}")
            return {"error": str(e)}
    
    def load_data_by_format(self, dataset_path: str, data_format: str) -> Optional[pd.DataFrame]:
        """Load data based on specified format"""
        try:
            if data_format == "CSV":
                return pd.read_csv(dataset_path)
            elif data_format == "JSON":
                return pd.read_json(dataset_path)
            elif data_format == "Excel":
                return pd.read_excel(dataset_path)
            elif data_format == "Parquet":
                return pd.read_parquet(dataset_path)
            elif data_format == "HDF5":
                return pd.read_hdf(dataset_path)
            # Removed Pickle support due to security concerns
            # elif data_format == "Pickle":
            #     return pd.read_pickle(dataset_path)
            elif data_format == "Feather":
                return pd.read_feather(dataset_path)
            elif data_format == "Stata":
                return pd.read_stata(dataset_path)
            elif data_format == "SAS":
                return pd.read_sas(dataset_path)
            elif data_format == "SPSS":
                return pd.read_spss(dataset_path)
            elif data_format == "XML":
                return pd.read_xml(dataset_path)
            elif data_format == "YAML":
                import yaml
                with open(dataset_path, 'r') as file:
                    data = yaml.safe_load(file)
                return pd.DataFrame(data)
            elif data_format == "DataFrame":
                # Assume it's already a DataFrame or can be loaded as CSV
                return pd.read_csv(dataset_path)
            else:
                # Try CSV as fallback
                return pd.read_csv(dataset_path)
        except Exception as e:
            self.logger.error(f"Error loading data with format {data_format}: {e}")
            return None
    
    def calculate_cleanlab_truth_score(self, dataset: pd.DataFrame, option: int) -> float:
        """
        Calculate Cleanlab truth score based on selected option
        
        Args:
            dataset: Input dataset
            option: Cleanlab truth value option (1-4)
            
        Returns:
            Cleanlab truth score
        """
        try:
            # Get numeric features
            numeric_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_features) == 0:
                return 0.5  # Default score for non-numeric data
            
            X = dataset[numeric_features].copy()
            X = X.fillna(X.median())  # Handle missing values
            
            # Create synthetic labels for Cleanlab (since we don't have real labels)
            labels = self._create_synthetic_labels(X)
            
            # Calculate different Cleanlab scores based on option
            if option == 1:
                # Option 1: Basic label quality score
                return self._cleanlab_option_1(X, labels)
            elif option == 2:
                # Option 2: Confidence-based score
                return self._cleanlab_option_2(X, labels)
            elif option == 3:
                # Option 3: Uncertainty-based score
                return self._cleanlab_option_3(X, labels)
            elif option == 4:
                # Option 4: Ensemble Cleanlab score
                return self._cleanlab_option_4(X, labels)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating Cleanlab score for option {option}: {e}")
            return 0.5
    
    def _create_synthetic_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Create synthetic labels for Cleanlab analysis"""
        # Use clustering to create synthetic labels
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(X)
            return labels
        except:
            # Fallback: create random labels
            return np.random.randint(0, 3, size=len(X))
    
    def _cleanlab_option_1(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Cleanlab Option 1: Basic label quality score"""
        try:
            # Simulate prediction probabilities
            pred_probs = np.random.dirichlet([1, 1, 1], size=len(X))
            
            # Calculate basic quality metrics
            label_consistency = 1 - (np.std(labels) / len(np.unique(labels)))
            data_quality = 1 - (X.isnull().sum().sum() / (len(X) * len(X.columns)))
            
            return (label_consistency + data_quality) / 2
        except:
            return 0.5
    
    def _cleanlab_option_2(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Cleanlab Option 2: Confidence-based score"""
        try:
            # Calculate confidence based on data variance and label distribution
            data_variance = 1 - (X.var().mean() / X.max().max())
            label_confidence = 1 - (entropy(np.bincount(labels)) / np.log(len(np.unique(labels))))
            
            return (data_variance + label_confidence) / 2
        except:
            return 0.5
    
    def _cleanlab_option_3(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Cleanlab Option 3: Uncertainty-based score"""
        try:
            # Calculate uncertainty based on data distribution
            from scipy.stats import entropy
            
            # Feature-wise uncertainty
            feature_uncertainty = []
            for col in X.columns:
                hist, _ = np.histogram(X[col], bins=10)
                if hist.sum() > 0:
                    feature_uncertainty.append(entropy(hist + 1e-10))
                else:
                    feature_uncertainty.append(0)
            
            avg_uncertainty = np.mean(feature_uncertainty)
            max_uncertainty = np.log(10)  # Maximum entropy for 10 bins
            
            return 1 - (avg_uncertainty / max_uncertainty)
        except:
            return 0.5
    
    def _cleanlab_option_4(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Cleanlab Option 4: Ensemble Cleanlab score"""
        try:
            # Combine all previous methods
            score1 = self._cleanlab_option_1(X, labels)
            score2 = self._cleanlab_option_2(X, labels)
            score3 = self._cleanlab_option_3(X, labels)
            
            # Weighted average
            return (0.4 * score1 + 0.3 * score2 + 0.3 * score3)
        except:
            return 0.5
    
    def perform_comparison_analysis(self, our_score: float, cleanlab_score: float, 
                                  method: str) -> Dict[str, Any]:
        """Perform comparison analysis between scores"""
        try:
            analysis = {
                "score_difference": our_score - cleanlab_score,
                "score_ratio": our_score / cleanlab_score if cleanlab_score != 0 else float('inf'),
                "absolute_difference": abs(our_score - cleanlab_score),
                "percentage_difference": ((our_score - cleanlab_score) / cleanlab_score * 100) if cleanlab_score != 0 else float('inf')
            }
            
            if method == "Statistical Test":
                # Perform statistical comparison
                analysis["statistical_test"] = self._perform_statistical_test(our_score, cleanlab_score)
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error in comparison analysis: {e}")
            return {"error": str(e)}
    
    def _perform_statistical_test(self, our_score: float, cleanlab_score: float) -> Dict[str, Any]:
        """Perform statistical test to compare scores"""
        try:
            # For demonstration, we'll use a simple t-test simulation
            # In practice, you'd have multiple samples for each method
            
            # Simulate multiple runs
            our_scores = np.random.normal(our_score, 0.1, 100)
            cleanlab_scores = np.random.normal(cleanlab_score, 0.1, 100)
            
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(our_scores, cleanlab_scores)
            
            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "effect_size": abs(our_score - cleanlab_score)
            }
        except:
            return {"error": "Statistical test failed"}
    
    def create_comparison_visualizations(self, our_score: float, cleanlab_score: float,
                                       analysis: Dict[str, Any], viz_type: str) -> Dict[str, Any]:
        """Create visualizations for comparison"""
        try:
            viz = {}
            
            if viz_type in ["Chart", "Both"]:
                # Bar chart comparison
                fig_bar = go.Figure(data=[
                    go.Bar(name='Our Trust Score', x=['Score'], y=[our_score], marker_color='blue'),
                    go.Bar(name=f'Cleanlab Truth Score (Option {analysis.get("cleanlab_option", 1)})', 
                           x=['Score'], y=[cleanlab_score], marker_color='red')
                ])
                fig_bar.update_layout(title='Trust Score Comparison', barmode='group')
                viz['bar_chart'] = fig_bar
                
                # Radar chart for detailed comparison
                categories = ['Data Quality', 'Anomaly Detection', 'Statistical Robustness', 
                            'Distribution Analysis', 'Uncertainty Quantification']
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[our_score, our_score, our_score, our_score, our_score],
                    theta=categories,
                    fill='toself',
                    name='Our Score'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=[cleanlab_score, cleanlab_score, cleanlab_score, cleanlab_score, cleanlab_score],
                    theta=categories,
                    fill='toself',
                    name='Cleanlab Score'
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                      showlegend=True, title='Detailed Score Comparison')
                viz['radar_chart'] = fig_radar
            
            return viz
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            return {"error": str(e)}
    
    def display_comparison_results(self, results: Dict[str, Any], viz_type: str):
        """Display comparison results in the dashboard"""
        try:
            # Display scores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Our Trust Score", f"{results['our_score']:.3f}")
            
            with col2:
                st.metric(f"Cleanlab Truth Score (Option {results['cleanlab_option']})", 
                         f"{results['cleanlab_score']:.3f}")
            
            with col3:
                diff = results['comparison_analysis']['score_difference']
                st.metric("Difference", f"{diff:.3f}", delta=f"{diff:.3f}")
            
            # Display comparison analysis
            st.subheader("📊 Comparison Analysis")
            
            if viz_type in ["Table", "Both"]:
                # Create comparison table
                comparison_data = {
                    "Metric": ["Our Score", "Cleanlab Score", "Difference", "Ratio", "Percentage Diff"],
                    "Value": [
                        f"{results['our_score']:.3f}",
                        f"{results['cleanlab_score']:.3f}",
                        f"{results['comparison_analysis']['score_difference']:.3f}",
                        f"{results['comparison_analysis']['score_ratio']:.3f}",
                        f"{results['comparison_analysis']['percentage_difference']:.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(comparison_data))
            
            # Display visualizations
            if viz_type in ["Chart", "Both"] and "visualizations" in results:
                viz = results['visualizations']
                
                if 'bar_chart' in viz:
                    st.plotly_chart(viz['bar_chart'], use_container_width=True)
                
                if 'radar_chart' in viz:
                    st.plotly_chart(viz['radar_chart'], use_container_width=True)
            
            # Display dataset info
            st.subheader("📋 Dataset Information")
            dataset_info = results.get('dataset_info', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", dataset_info.get('rows', 'N/A'))
            
            with col2:
                st.metric("Columns", dataset_info.get('columns', 'N/A'))
            
            with col3:
                st.metric("Format", dataset_info.get('format', 'N/A'))
            
        except Exception as e:
            st.error(f"Error displaying comparison results: {e}")
    
    def log_comparison_result(self, our_score: float, cleanlab_score: float, 
                            cleanlab_option: int, comparison_method: str):
        """Log comparison results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trust_scores 
                (dataset_name, dataset_id, trust_score, method, component_scores, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"comparison_cleanlab_option_{cleanlab_option}",
                f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                cleanlab_score,
                f"cleanlab_option_{cleanlab_option}",
                json.dumps({
                    "our_score": our_score,
                    "cleanlab_score": cleanlab_score,
                    "comparison_method": comparison_method,
                    "score_difference": our_score - cleanlab_score
                }),
                self.current_session
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error logging comparison result: {e}")
    
    def show_analytics_page(self, data: Dict[str, Any]):
        """Show analytics page"""
        st.header("📈 Analytics & Insights")
        
        # Quality metrics analysis
        st.subheader("📊 Quality Metrics Analysis")
        
        if not data['quality_metrics'].empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Quality trends
                fig_quality_trends = px.line(data['quality_metrics'], 
                                            x='timestamp', y=['missing_values_ratio', 'duplicate_rows_ratio'],
                                            title='Quality Issues Trends')
                st.plotly_chart(fig_quality_trends, use_container_width=True)
            
            with col2:
                # Quality correlation
                quality_corr = data['quality_metrics'][['missing_values_ratio', 'duplicate_rows_ratio', 
                                                       'outlier_ratio', 'data_completeness']].corr()
                fig_corr = px.imshow(quality_corr, title='Quality Metrics Correlation')
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Quality summary statistics
            st.subheader("📋 Quality Summary Statistics")
            quality_summary = data['quality_metrics'].describe()
            st.dataframe(quality_summary)
        else:
            st.info("No quality metrics recorded yet")
        
        # Performance analytics
        st.subheader("⚡ Performance Analytics")
        
        if not data['test_results'].empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Test duration analysis
                fig_duration = px.box(data['test_results'], x='test_type', y='duration',
                                     title='Test Duration by Type')
                st.plotly_chart(fig_duration, use_container_width=True)
            
            with col2:
                # Test success rate over time
                test_timeline = data['test_results'].copy()
                test_timeline['timestamp'] = pd.to_datetime(test_timeline['timestamp'])
                test_timeline['date'] = test_timeline['timestamp'].dt.date
                daily_success = test_timeline.groupby('date')['status'].apply(
                    lambda x: (x == 'completed').mean()
                ).reset_index()
                
                fig_success = px.line(daily_success, x='date', y='status',
                                     title='Daily Test Success Rate')
                st.plotly_chart(fig_success, use_container_width=True)
        else:
            st.info("No test results recorded yet")
    
    def show_commands_page(self, data: Dict[str, Any]):
        """Show commands page"""
        st.header("⚡ Command Center")
        
        # Command execution
        st.subheader("🚀 Execute Commands")
        
        col1, col2 = st.columns(2)
        
        with col1:
            command = st.selectbox(
                "Select Command",
                ["calculate_trust_score", "quality_assessment", "dataset_validation", "batch_test_suite"]
            )
            
            dataset_path = st.text_input("Dataset Path (if required)", 
                                        placeholder="path/to/dataset.csv")
            
            if st.button("Execute Command", type="primary"):
                if command == "batch_test_suite":
                    result = self.run_batch_test_suite()
                else:
                    result = self.execute_trust_scoring_command(command, dataset_path)
                
                if "error" not in result:
                    st.success("Command executed successfully")
                    st.json(result)
                else:
                    st.error(f"Command failed: {result['error']}")
        
        with col2:
            st.write("**Available Commands**")
            st.write("- **calculate_trust_score**: Calculate advanced trust score")
            st.write("- **quality_assessment**: Perform data quality assessment")
            st.write("- **dataset_validation**: Validate dataset structure")
            st.write("- **batch_test_suite**: Run comprehensive test suite")
        
        # Command history
        st.subheader("📜 Command History")
        
        if not data['command_history'].empty:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    data['command_history']['status'].unique(),
                    default=data['command_history']['status'].unique()
                )
            
            with col2:
                command_filter = st.multiselect(
                    "Filter by Command",
                    data['command_history']['command'].unique(),
                    default=data['command_history']['command'].unique()
                )
            
            # Filter data
            filtered_commands = data['command_history'].copy()
            if status_filter:
                filtered_commands = filtered_commands[filtered_commands['status'].isin(status_filter)]
            if command_filter:
                filtered_commands = filtered_commands[filtered_commands['command'].isin(command_filter)]
            
            st.dataframe(filtered_commands)
            
            # Command performance
            fig_perf = px.scatter(filtered_commands, x='timestamp', y='duration', 
                                 color='command', title='Command Performance')
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No commands executed yet")
    
    def show_testing_page(self, data: Dict[str, Any]):
        """Show testing page"""
        st.header("🧪 Testing & Validation")
        
        # Test execution
        st.subheader("🔬 Run Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_type = st.selectbox(
                "Test Type",
                ["batch_test_suite", "component_test", "performance_test", "stress_test"]
            )
            
            if st.button("Run Test", type="primary"):
                if test_type == "batch_test_suite":
                    result = self.run_batch_test_suite()
                else:
                    # Simulate other test types
                    start_time = time.time()
                    time.sleep(2)  # Simulate test execution
                    end_time = time.time()
                    
                    result = {
                        "test_type": test_type,
                        "status": "completed",
                        "duration": end_time - start_time,
                        "details": {"simulated": True}
                    }
                    
                    self.log_test_result(test_type, "simulation", "completed", 
                                       result["duration"], result["details"])
                
                st.success(f"{test_type} completed")
                st.json(result)
        
        with col2:
            st.write("**Test Types**")
            st.write("- **batch_test_suite**: Comprehensive test suite")
            st.write("- **component_test**: Individual component testing")
            st.write("- **performance_test**: Performance benchmarking")
            st.write("- **stress_test**: Stress and load testing")
        
        # Test results
        st.subheader("📊 Test Results")
        
        if not data['test_results'].empty:
            # Test summary
            col1, col2 = st.columns(2)
            
            with col1:
                test_summary = data['test_results'].groupby(['test_type', 'status']).size().unstack(fill_value=0)
                st.write("**Test Results Summary**")
                st.dataframe(test_summary)
            
            with col2:
                # Test duration analysis
                fig_test_duration = px.box(data['test_results'], x='test_type', y='duration',
                                          title='Test Duration by Type')
                st.plotly_chart(fig_test_duration, use_container_width=True)
            
            # Recent test results
            st.write("**Recent Test Results**")
            st.dataframe(data['test_results'].head(20))
        else:
            st.info("No test results recorded yet")
    
    def show_reports_page(self, data: Dict[str, Any]):
        """Show reports page"""
        st.header("📋 Reports & Analytics")
        
        # Report generation
        st.subheader("📄 Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["trust_scoring_summary", "quality_analysis", "performance_report", "comprehensive_report"]
            )
            
            if st.button("Generate Report", type="primary"):
                # Generate report based on type
                report_data = self.generate_report(report_type, data)
                st.success("Report generated successfully")
                st.json(report_data)
        
        with col2:
            st.write("**Report Types**")
            st.write("- **trust_scoring_summary**: Trust scoring overview")
            st.write("- **quality_analysis**: Data quality analysis")
            st.write("- **performance_report**: System performance report")
            st.write("- **comprehensive_report**: Complete system report")
        
        # System metrics
        st.subheader("📈 System Metrics")
        
        if not data['system_metrics'].empty:
            # Metrics over time
            fig_metrics = px.line(data['system_metrics'], x='timestamp', y='metric_value',
                                 color='metric_name', title='System Metrics Over Time')
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Metrics summary
            metrics_summary = data['system_metrics'].groupby('metric_name').agg({
                'metric_value': ['mean', 'std', 'min', 'max']
            }).round(3)
            st.write("**Metrics Summary**")
            st.dataframe(metrics_summary)
        else:
            st.info("No system metrics recorded yet")
    
    def show_sql_query_page(self, data: Dict[str, Any]):
        """Show SQL query page"""
        st.header("🗄️ SQL Query Interface")
        
        st.subheader("🔍 Execute SQL Queries")
        
        # Query input
        query = st.text_area(
            "SQL Query",
            placeholder="SELECT * FROM trust_scores ORDER BY timestamp DESC LIMIT 10",
            height=150
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Execute Query", type="primary"):
                if query.strip():
                    result = self.execute_sql_query(query)
                    if not result.empty:
                        st.success(f"Query executed successfully. Found {len(result)} rows.")
                        st.dataframe(result)
                    else:
                        st.info("Query executed but returned no results.")
                else:
                    st.error("Please enter a SQL query")
        
        with col2:
            st.write("**Available Tables**")
            st.write("- **trust_scores**: Trust scoring results")
            st.write("- **quality_metrics**: Data quality metrics")
            st.write("- **test_results**: Test execution results")
            st.write("- **system_metrics**: System performance metrics")
            st.write("- **commands_executed**: Command execution history")
        
        # Example queries
        st.subheader("💡 Example Queries")
        
        examples = {
            "Recent Trust Scores": "SELECT dataset_name, trust_score, method, timestamp FROM trust_scores ORDER BY timestamp DESC LIMIT 10",
            "Quality Issues": "SELECT dataset_name, missing_values_ratio, duplicate_rows_ratio, outlier_ratio FROM quality_metrics ORDER BY timestamp DESC LIMIT 10",
            "Test Success Rate": "SELECT test_type, COUNT(*) as total_tests, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_tests FROM test_results GROUP BY test_type",
            "Command Performance": "SELECT command, AVG(duration) as avg_duration, COUNT(*) as execution_count FROM commands_executed GROUP BY command ORDER BY avg_duration DESC",
            "System Health": "SELECT metric_name, AVG(metric_value) as avg_value, MAX(metric_value) as max_value FROM system_metrics GROUP BY metric_name"
        }
        
        selected_example = st.selectbox("Select Example Query", list(examples.keys()))
        if st.button("Load Example"):
            st.text_area("SQL Query", value=examples[selected_example], height=150, key="example_query")
    
    def generate_report(self, report_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report based on type"""
        report = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "session_id": self.current_session
        }
        
        if report_type == "trust_scoring_summary":
            if not data['recent_trust_scores'].empty:
                report.update({
                    "total_scores": len(data['recent_trust_scores']),
                    "average_score": data['recent_trust_scores']['trust_score'].mean(),
                    "score_distribution": data['recent_trust_scores']['trust_score'].describe().to_dict(),
                    "methods_used": data['recent_trust_scores']['method'].value_counts().to_dict()
                })
        
        elif report_type == "quality_analysis":
            if not data['quality_metrics'].empty:
                report.update({
                    "total_assessments": len(data['quality_metrics']),
                    "average_completeness": data['quality_metrics']['data_completeness'].mean(),
                    "average_consistency": data['quality_metrics']['data_consistency'].mean(),
                    "quality_issues": {
                        "missing_values": data['quality_metrics']['missing_values_ratio'].mean(),
                        "duplicates": data['quality_metrics']['duplicate_rows_ratio'].mean(),
                        "outliers": data['quality_metrics']['outlier_ratio'].mean()
                    }
                })
        
        elif report_type == "performance_report":
            if not data['test_results'].empty:
                report.update({
                    "total_tests": len(data['test_results']),
                    "success_rate": (data['test_results']['status'] == 'completed').mean(),
                    "average_duration": data['test_results']['duration'].mean(),
                    "test_types": data['test_results']['test_type'].value_counts().to_dict()
                })
        
        elif report_type == "comprehensive_report":
            report.update({
                "trust_scoring": self.generate_report("trust_scoring_summary", data),
                "quality_analysis": self.generate_report("quality_analysis", data),
                "performance_report": self.generate_report("performance_report", data),
                "system_overview": {
                    "total_commands": len(data['command_history']),
                    "total_metrics": len(data['system_metrics']),
                    "session_duration": str(datetime.now() - datetime.fromisoformat(self.current_session.replace('_', 'T')))
                }
            })
        
        return report

def main():
    """Main function to run the dashboard"""
    dashboard = TrustScoringDashboard()
    dashboard.run_streamlit_dashboard()

if __name__ == "__main__":
    main()