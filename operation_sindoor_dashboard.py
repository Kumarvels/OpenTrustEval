import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List
import asyncio

class OperationSindoorDashboard:
    """Specialized dashboard for Operation Sindoor analysis and reporting"""
    
    def __init__(self):
        self.report_file = "operation_sindoor_test_report_20250713_134117.json"
        self.load_report_data()
    
    def load_report_data(self):
        """Load the Operation Sindoor test report data"""
        try:
            with open(self.report_file, 'r') as f:
                self.report_data = json.load(f)
            self.process_data()
        except FileNotFoundError:
            st.error(f"Report file {self.report_file} not found!")
            self.report_data = None
        except Exception as e:
            st.error(f"Error loading report: {e}")
            self.report_data = None
    
    def process_data(self):
        """Process the report data for visualization"""
        if not self.report_data:
            return
        
        # Convert detailed results to DataFrame
        self.df = pd.DataFrame(self.report_data['detailed_results'])
        
        # Add derived columns
        self.df['test_category'] = self.df['test_name'].apply(self.categorize_test)
        self.df['detection_status'] = self.df['correct_detection'].apply(
            lambda x: 'Correct' if x else 'Incorrect'
        )
        self.df['misinformation_type'] = self.df['test_name'].apply(self.extract_misinfo_type)
        
        # Calculate summary statistics
        self.summary_stats = self.report_data['test_summary']
    
    def categorize_test(self, test_name: str) -> str:
        """Categorize test based on name"""
        if 'Legitimate' in test_name:
            return 'Legitimate News'
        elif 'Misinformation' in test_name:
            return 'Misinformation'
        elif 'Disinformation' in test_name:
            return 'Disinformation'
        elif 'Mixed Truth' in test_name:
            return 'Mixed Truth'
        elif 'Conspiracy' in test_name:
            return 'Conspiracy Theory'
        elif 'Technical' in test_name:
            return 'Technical Misinformation'
        elif 'Factual' in test_name:
            return 'Factual Information'
        else:
            return 'Other'
    
    def extract_misinfo_type(self, test_name: str) -> str:
        """Extract specific misinformation type"""
        if 'False Casualty' in test_name:
            return 'False Casualty Numbers'
        elif 'False Timeline' in test_name:
            return 'False Timeline'
        elif 'False International' in test_name:
            return 'False International Involvement'
        elif 'False Nuclear' in test_name:
            return 'False Nuclear Threat'
        elif 'Exaggerated Claims' in test_name:
            return 'Exaggerated Claims'
        elif 'Speculative Analysis' in test_name:
            return 'Speculative Analysis'
        elif 'Hidden Agenda' in test_name:
            return 'Hidden Agenda'
        elif 'False Equipment' in test_name:
            return 'False Equipment'
        else:
            return 'Other'
    
    def render_dashboard(self):
        """Render the Operation Sindoor dashboard"""
        st.set_page_config(
            page_title="Operation Sindoor Analysis Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎯 Operation Sindoor Analysis Dashboard")
        st.markdown("### Real-time Misinformation Detection & Analysis Report")
        
        if not self.report_data:
            st.error("No report data available. Please ensure the report file exists.")
            return
        
        # Sidebar controls
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Analysis View",
            ["📊 Executive Summary", "🔍 Detailed Analysis", "📈 Performance Metrics", 
             "🎯 Detection Accuracy", "⚡ Real-time Monitoring", "📋 Full Report"]
        )
        
        # Page routing
        if page == "📊 Executive Summary":
            self.render_executive_summary()
        elif page == "🔍 Detailed Analysis":
            self.render_detailed_analysis()
        elif page == "📈 Performance Metrics":
            self.render_performance_metrics()
        elif page == "🎯 Detection Accuracy":
            self.render_detection_accuracy()
        elif page == "⚡ Real-time Monitoring":
            self.render_realtime_monitoring()
        elif page == "📋 Full Report":
            self.render_full_report()
    
    def render_executive_summary(self):
        """Render executive summary page"""
        st.header("📊 Executive Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = self.summary_stats['accuracy'] * 100
            st.metric(
                "Overall Accuracy", 
                f"{accuracy:.1f}%",
                f"{'📈' if accuracy >= 80 else '📉'}"
            )
        
        with col2:
            correct_detections = self.summary_stats['correct_detections']
            total_tests = self.summary_stats['total_tests']
            st.metric(
                "Correct Detections", 
                f"{correct_detections}/{total_tests}",
                f"{'✅' if correct_detections >= total_tests/2 else '⚠️'}"
            )
        
        with col3:
            avg_latency = self.summary_stats['avg_latency_ms']
            st.metric(
                "Average Latency", 
                f"{avg_latency:.0f}ms",
                f"{'⚡' if avg_latency < 100 else '🐌'}"
            )
        
        with col4:
            avg_confidence = self.summary_stats['avg_confidence'] * 100
            st.metric(
                "Average Confidence", 
                f"{avg_confidence:.1f}%",
                f"{'🎯' if avg_confidence >= 80 else '❓'}"
            )
        
        # Summary insights
        st.subheader("🎯 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Detection Performance**")
            st.write(f"• **Accuracy**: {accuracy:.1f}% overall detection accuracy")
            st.write(f"• **Misinformation Detection**: {self.summary_stats['misinformation_detection_rate']*100:.1f}% rate")
            st.write(f"• **Legitimate Identification**: {self.summary_stats['legitimate_identification_rate']*100:.1f}% rate")
            st.write(f"• **Average Hallucination Risk**: {self.summary_stats['avg_hallucination_risk']*100:.1f}%")
        
        with col2:
            st.success("**System Performance**")
            st.write(f"• **Response Time**: {avg_latency:.0f}ms average latency")
            st.write(f"• **Confidence Level**: {avg_confidence:.1f}% average confidence")
            st.write(f"• **Test Coverage**: {total_tests} comprehensive test scenarios")
            st.write(f"• **Domain Coverage**: Government, Media, Technology domains")
        
        # Quick visualization
        st.subheader("📊 Performance Overview")
        
        # Accuracy by category
        category_accuracy = self.df.groupby('test_category')['correct_detection'].agg(['count', 'sum']).reset_index()
        category_accuracy['accuracy'] = category_accuracy['sum'] / category_accuracy['count'] * 100
        
        fig = px.bar(
            category_accuracy, 
            x='test_category', 
            y='accuracy',
            title="Detection Accuracy by Test Category",
            labels={'accuracy': 'Accuracy (%)', 'test_category': 'Test Category'},
            color='accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_analysis(self):
        """Render detailed analysis page"""
        st.header("🔍 Detailed Analysis")
        
        # Test results table
        st.subheader("📋 Test Results Summary")
        
        # Create enhanced dataframe for display
        display_df = self.df[['test_name', 'test_category', 'domain', 'correct_detection', 
                             'confidence', 'hallucination_risk', 'latency_ms', 'source_accuracy']].copy()
        display_df['confidence'] = display_df['confidence'] * 100
        display_df['hallucination_risk'] = display_df['hallucination_risk'] * 100
        display_df['source_accuracy'] = display_df['source_accuracy'] * 100
        display_df['correct_detection'] = display_df['correct_detection'].apply(
            lambda x: '✅ Correct' if x else '❌ Incorrect'
        )
        
        st.dataframe(display_df, use_container_width=True)
        
        # Detailed analysis by category
        st.subheader("📊 Analysis by Category")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = self.df['test_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Test Distribution by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Domain distribution
            domain_counts = self.df['domain'].value_counts()
            fig = px.bar(
                x=domain_counts.index,
                y=domain_counts.values,
                title="Test Distribution by Domain",
                labels={'x': 'Domain', 'y': 'Number of Tests'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed test analysis
        st.subheader("🔬 Individual Test Analysis")
        
        # Select test for detailed view
        selected_test = st.selectbox(
            "Select Test for Detailed Analysis",
            self.df['test_name'].tolist()
        )
        
        if selected_test:
            test_data = self.df[self.df['test_name'] == selected_test].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Detection Status", 
                         "✅ Correct" if test_data['correct_detection'] else "❌ Incorrect")
                st.metric("Domain", test_data['domain'].title())
                st.metric("Category", test_data['test_category'])
            
            with col2:
                st.metric("Confidence", f"{test_data['confidence']*100:.1f}%")
                st.metric("Hallucination Risk", f"{test_data['hallucination_risk']*100:.1f}%")
                st.metric("Latency", f"{test_data['latency_ms']:.0f}ms")
            
            with col3:
                st.metric("Source Accuracy", f"{test_data['source_accuracy']*100:.1f}%")
                st.metric("Verified", "✅ Yes" if test_data['verified'] else "❌ No")
                st.metric("Misinformation", "✅ Yes" if test_data['is_misinformation'] else "❌ No")
            
            # Sources used
            st.subheader("📚 Sources Used")
            sources_used = test_data['sources_used']
            expected_sources = test_data['expected_sources']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Actual Sources Used:**")
                for source in sources_used:
                    st.write(f"• {source}")
            
            with col2:
                st.write("**Expected Sources:**")
                for source in expected_sources:
                    st.write(f"• {source}")
    
    def render_performance_metrics(self):
        """Render performance metrics page"""
        st.header("📈 Performance Metrics")
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Latency", f"{self.summary_stats['avg_latency_ms']:.0f}ms")
        with col2:
            st.metric("Average Confidence", f"{self.summary_stats['avg_confidence']*100:.1f}%")
        with col3:
            st.metric("Average Hallucination Risk", f"{self.summary_stats['avg_hallucination_risk']*100:.1f}%")
        with col4:
            st.metric("Source Accuracy", f"{self.df['source_accuracy'].mean()*100:.1f}%")
        
        # Performance charts
        st.subheader("📊 Performance Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Latency distribution
            fig = px.histogram(
                self.df, 
                x='latency_ms',
                title="Latency Distribution",
                labels={'latency_ms': 'Latency (ms)', 'count': 'Number of Tests'},
                nbins=10
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence vs Hallucination Risk
            fig = px.scatter(
                self.df,
                x='confidence',
                y='hallucination_risk',
                color='correct_detection',
                title="Confidence vs Hallucination Risk",
                labels={'confidence': 'Confidence', 'hallucination_risk': 'Hallucination Risk'},
                color_discrete_map={True: 'green', False: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance by domain
        st.subheader("🎯 Performance by Domain")
        
        domain_performance = self.df.groupby('domain').agg({
            'correct_detection': ['count', 'sum'],
            'latency_ms': 'mean',
            'confidence': 'mean',
            'hallucination_risk': 'mean',
            'source_accuracy': 'mean'
        }).round(3)
        
        domain_performance.columns = ['Total Tests', 'Correct Detections', 'Avg Latency (ms)', 
                                    'Avg Confidence', 'Avg Hallucination Risk', 'Avg Source Accuracy']
        domain_performance['Accuracy %'] = (domain_performance['Correct Detections'] / 
                                           domain_performance['Total Tests'] * 100).round(1)
        
        st.dataframe(domain_performance, use_container_width=True)
        
        # Performance heatmap
        st.subheader("🔥 Performance Heatmap")
        
        # Create heatmap data
        heatmap_data = self.df.pivot_table(
            values='correct_detection',
            index='domain',
            columns='test_category',
            aggfunc='mean'
        ).fillna(0) * 100
        
        fig = px.imshow(
            heatmap_data,
            title="Detection Accuracy Heatmap (Domain vs Category)",
            labels=dict(x="Test Category", y="Domain", color="Accuracy (%)"),
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_detection_accuracy(self):
        """Render detection accuracy analysis"""
        st.header("🎯 Detection Accuracy Analysis")
        
        # Overall accuracy breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy pie chart
            correct_count = self.summary_stats['correct_detections']
            incorrect_count = self.summary_stats['total_tests'] - correct_count
            
            fig = px.pie(
                values=[correct_count, incorrect_count],
                names=['Correct Detections', 'Incorrect Detections'],
                title="Overall Detection Accuracy",
                color_discrete_sequence=['green', 'red']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accuracy by category
            category_accuracy = self.df.groupby('test_category')['correct_detection'].agg(['count', 'sum']).reset_index()
            category_accuracy['accuracy'] = category_accuracy['sum'] / category_accuracy['count'] * 100
            
            fig = px.bar(
                category_accuracy,
                x='test_category',
                y='accuracy',
                title="Accuracy by Test Category",
                labels={'accuracy': 'Accuracy (%)', 'test_category': 'Test Category'},
                color='accuracy',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Misinformation detection analysis
        st.subheader("🚨 Misinformation Detection Analysis")
        
        # Filter misinformation tests
        misinfo_tests = self.df[self.df['is_misinformation'] == True]
        
        if not misinfo_tests.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Misinformation detection accuracy
                misinfo_accuracy = (misinfo_tests['correct_detection'].sum() / 
                                   len(misinfo_tests) * 100)
                
                st.metric("Misinformation Detection Accuracy", f"{misinfo_accuracy:.1f}%")
                
                # Misinformation types
                misinfo_types = misinfo_tests['misinformation_type'].value_counts()
                fig = px.pie(
                    values=misinfo_types.values,
                    names=misinfo_types.index,
                    title="Misinformation Types Detected"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Misinformation by domain
                misinfo_by_domain = misinfo_tests.groupby('domain')['correct_detection'].agg(['count', 'sum']).reset_index()
                misinfo_by_domain['accuracy'] = misinfo_by_domain['sum'] / misinfo_by_domain['count'] * 100
                
                fig = px.bar(
                    misinfo_by_domain,
                    x='domain',
                    y='accuracy',
                    title="Misinformation Detection by Domain",
                    labels={'accuracy': 'Detection Accuracy (%)', 'domain': 'Domain'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No misinformation tests found in the dataset.")
        
        # False positive/negative analysis
        st.subheader("🔍 False Positive/Negative Analysis")
        
        # Calculate confusion matrix
        legitimate_tests = self.df[self.df['is_misinformation'] == False]
        misinfo_tests = self.df[self.df['is_misinformation'] == True]
        
        if not legitimate_tests.empty and not misinfo_tests.empty:
            # True positives (correctly identified misinformation)
            tp = misinfo_tests['correct_detection'].sum()
            # False negatives (misinformation not detected)
            fn = len(misinfo_tests) - tp
            # True negatives (correctly identified legitimate)
            tn = legitimate_tests['correct_detection'].sum()
            # False positives (legitimate marked as misinformation)
            fp = len(legitimate_tests) - tn
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("True Positives", tp, "Correctly detected misinformation")
            with col2:
                st.metric("False Negatives", fn, "Missed misinformation")
            with col3:
                st.metric("True Negatives", tn, "Correctly identified legitimate")
            with col4:
                st.metric("False Positives", fp, "Legitimate marked as misinformation")
            
            # Confusion matrix visualization
            confusion_data = pd.DataFrame({
                'Actual': ['Misinformation', 'Misinformation', 'Legitimate', 'Legitimate'],
                'Predicted': ['Misinformation', 'Legitimate', 'Misinformation', 'Legitimate'],
                'Count': [tp, fn, fp, tn],
                'Type': ['True Positive', 'False Negative', 'False Positive', 'True Negative']
            })
            
            fig = px.bar(
                confusion_data,
                x='Type',
                y='Count',
                title="Confusion Matrix Breakdown",
                color='Type',
                color_discrete_map={
                    'True Positive': 'green',
                    'True Negative': 'blue',
                    'False Positive': 'orange',
                    'False Negative': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_realtime_monitoring(self):
        """Render real-time monitoring page"""
        st.header("⚡ Real-time Monitoring")
        
        st.info("This section would show real-time monitoring of the Operation Sindoor verification system.")
        
        # Simulated real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Requests", "12", "+3")
        with col2:
            st.metric("System Load", "78%", "+5%")
        with col3:
            st.metric("Response Time", "85ms", "-12ms")
        with col4:
            st.metric("Success Rate", "96.2%", "+1.2%")
        
        # Real-time activity feed
        st.subheader("📡 Real-time Activity Feed")
        
        # Simulate real-time data
        import time
        current_time = datetime.now().strftime("%H:%M:%S")
        
        activity_data = [
            {"time": current_time, "event": "New verification request received", "status": "Processing"},
            {"time": current_time, "event": "Government domain expert activated", "status": "Active"},
            {"time": current_time, "event": "Cross-domain verification completed", "status": "Success"},
            {"time": current_time, "event": "Hallucination risk assessment", "status": "Low Risk"},
            {"time": current_time, "event": "Response sent to client", "status": "Completed"}
        ]
        
        for activity in activity_data:
            status_color = {
                "Processing": "🟡",
                "Active": "🟢", 
                "Success": "✅",
                "Low Risk": "🟢",
                "Completed": "✅"
            }.get(activity["status"], "⚪")
            
            st.write(f"**{activity['time']}** {status_color} {activity['event']} - {activity['status']}")
        
        # System health indicators
        st.subheader("🏥 System Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU Usage
            cpu_usage = 78
            st.metric("CPU Usage", f"{cpu_usage}%")
            st.progress(cpu_usage / 100)
            
            # Memory Usage
            memory_usage = 65
            st.metric("Memory Usage", f"{memory_usage}%")
            st.progress(memory_usage / 100)
        
        with col2:
            # Network Latency
            network_latency = 45
            st.metric("Network Latency", f"{network_latency}ms")
            
            # Expert Utilization
            expert_utilization = 92
            st.metric("Expert Utilization", f"{expert_utilization}%")
            st.progress(expert_utilization / 100)
    
    def render_full_report(self):
        """Render full report page"""
        st.header("📋 Full Operation Sindoor Report")
        
        # Report metadata
        st.subheader("📄 Report Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Report File:**", self.report_file)
            st.write("**Total Tests:**", self.summary_stats['total_tests'])
            st.write("**Test Date:**", "July 13, 2025")
            st.write("**System Version:**", "Ultimate MoE v2.0")
        
        with col2:
            st.write("**Overall Accuracy:**", f"{self.summary_stats['accuracy']*100:.1f}%")
            st.write("**Average Latency:**", f"{self.summary_stats['avg_latency_ms']:.0f}ms")
            st.write("**Average Confidence:**", f"{self.summary_stats['avg_confidence']*100:.1f}%")
            st.write("**Report Generated:**", "Real-time")
        
        # Complete test results
        st.subheader("📊 Complete Test Results")
        
        # Enhanced display dataframe
        full_display_df = self.df.copy()
        full_display_df['confidence'] = full_display_df['confidence'] * 100
        full_display_df['hallucination_risk'] = full_display_df['hallucination_risk'] * 100
        full_display_df['source_accuracy'] = full_display_df['source_accuracy'] * 100
        full_display_df['correct_detection'] = full_display_df['correct_detection'].apply(
            lambda x: '✅ Correct' if x else '❌ Incorrect'
        )
        full_display_df['verified'] = full_display_df['verified'].apply(
            lambda x: '✅ Yes' if x else '❌ No'
        )
        full_display_df['is_misinformation'] = full_display_df['is_misinformation'].apply(
            lambda x: '✅ Yes' if x else '❌ No'
        )
        
        # Rename columns for better display
        full_display_df = full_display_df.rename(columns={
            'test_name': 'Test Name',
            'test_category': 'Category',
            'domain': 'Domain',
            'correct_detection': 'Detection Status',
            'confidence': 'Confidence (%)',
            'hallucination_risk': 'Hallucination Risk (%)',
            'latency_ms': 'Latency (ms)',
            'source_accuracy': 'Source Accuracy (%)',
            'verified': 'Verified',
            'is_misinformation': 'Is Misinformation'
        })
        
        st.dataframe(full_display_df, use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Full Report (JSON)"):
                st.download_button(
                    label="Click to download",
                    data=json.dumps(self.report_data, indent=2),
                    file_name="operation_sindoor_full_report.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download Analysis (CSV)"):
                csv_data = full_display_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv_data,
                    file_name="operation_sindoor_analysis.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Download Summary (PDF)"):
                st.info("PDF generation feature coming soon!")
        
        # Raw JSON view
        st.subheader("🔧 Raw Report Data")
        
        if st.checkbox("Show raw JSON data"):
            st.json(self.report_data)

# Run the dashboard
if __name__ == "__main__":
    dashboard = OperationSindoorDashboard()
    dashboard.render_dashboard() 