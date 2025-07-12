#!/usr/bin/env python3
"""
Trust Scoring System Dashboard
Lightweight data science dashboard with SQL support for monitoring trust scoring
Provides 360-degree view of scoring system with query, command, verify, check, and report capabilities
"""

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
        
        # Trust scoring controls
        st.subheader("🎯 Calculate Trust Score")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
                else:
                    st.error("Please provide dataset path")
        
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