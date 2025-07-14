#!/usr/bin/env python3
"""
Deployment Script for High-Performance Hallucination Detection System

This script provides automated deployment options for:
- Local development
- Docker containers
- Cloud platforms (AWS, GCP, Azure)
- Kubernetes clusters
"""

import os
import sys
import subprocess
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighPerformanceDeployer:
    """Deployment manager for the high-performance hallucination detection system"""
    
    def __init__(self, config_path: str = "deployment_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.project_root = Path(__file__).parent
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default deployment configuration"""
        config = {
            'system': {
                'name': 'high-performance-hallucination-detector',
                'version': '2.0.0',
                'port': 8002,
                'host': '0.0.0.0'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'password': None
            },
            'performance': {
                'max_workers': 50,
                'max_concurrent_requests': 100,
                'cache_ttl': 3600,
                'timeout': 5.0
            },
            'monitoring': {
                'enabled': True,
                'metrics_port': 9090,
                'health_check_interval': 30
            },
            'security': {
                'enable_auth': False,
                'enable_ssl': False,
                'rate_limit': 1000
            },
            'deployment': {
                'type': 'local',
                'environment': 'development'
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config

    def deploy_local(self):
        """Deploy locally for development"""
        logger.info("Deploying locally for development...")
        
        # Check prerequisites
        self._check_prerequisites()
        
        # Install dependencies
        self._install_dependencies()
        
        # Start Redis if not running
        self._start_redis()
        
        # Start the application
        self._start_application()
        
        logger.info("✅ Local deployment completed successfully!")
        logger.info(f"🌐 Application available at: http://localhost:{self.config['system']['port']}")
        logger.info(f"📊 API documentation at: http://localhost:{self.config['system']['port']}/docs")

    def deploy_docker(self):
        """Deploy using Docker"""
        logger.info("Deploying with Docker...")
        
        # Create Dockerfile
        self._create_dockerfile()
        
        # Create docker-compose.yml
        self._create_docker_compose()
        
        # Build and run containers
        self._run_docker_deployment()
        
        logger.info("✅ Docker deployment completed successfully!")

    def deploy_cloud(self, platform: str):
        """Deploy to cloud platform"""
        logger.info(f"Deploying to {platform}...")
        
        if platform == 'aws':
            self._deploy_aws()
        elif platform == 'gcp':
            self._deploy_gcp()
        elif platform == 'azure':
            self._deploy_azure()
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def _check_prerequisites(self):
        """Check system prerequisites"""
        logger.info("Checking prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            raise RuntimeError(f"Python 3.9+ required, found {python_version.major}.{python_version.minor}")
        
        # Check Redis
        try:
            import redis
            r = redis.Redis(host=self.config['redis']['host'], port=self.config['redis']['port'])
            r.ping()
            logger.info("✅ Redis connection successful")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
        
        # Check required packages
        required_packages = ['fastapi', 'uvicorn', 'numpy', 'pandas']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"⚠️ Missing packages: {missing_packages}")
            logger.info("Run: pip install -r requirements_high_performance.txt")

    def _install_dependencies(self):
        """Install system dependencies"""
        logger.info("Installing dependencies...")
        
        requirements_file = self.project_root / "requirements_high_performance.txt"
        if requirements_file.exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            logger.info("✅ Dependencies installed")
        else:
            logger.warning("⚠️ Requirements file not found")

    def _start_redis(self):
        """Start Redis server"""
        logger.info("Starting Redis...")
        
        try:
            # Check if Redis is already running
            import redis
            r = redis.Redis(host=self.config['redis']['host'], port=self.config['redis']['port'])
            r.ping()
            logger.info("✅ Redis already running")
            return
        except Exception:
            pass
        
        # Try to start Redis
        try:
            subprocess.run(["redis-server", "--daemonize", "yes"], check=True)
            logger.info("✅ Redis started")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ Could not start Redis automatically")
            logger.info("Please start Redis manually: redis-server")

    def _start_application(self):
        """Start the application"""
        logger.info("Starting application...")
        
        app_file = self.project_root / "high_performance_hallucination_detector.py"
        if not app_file.exists():
            raise FileNotFoundError(f"Application file not found: {app_file}")
        
        # Set environment variables
        env = os.environ.copy()
        env['REDIS_HOST'] = self.config['redis']['host']
        env['REDIS_PORT'] = str(self.config['redis']['port'])
        env['REDIS_DB'] = str(self.config['redis']['db'])
        
        # Start the application
        subprocess.run([
            sys.executable, str(app_file)
        ], env=env, check=True)

    def _create_dockerfile(self):
        """Create Dockerfile"""
        logger.info("Creating Dockerfile...")
        
        dockerfile_content = f"""
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_high_performance.txt .
RUN pip install --no-cache-dir -r requirements_high_performance.txt

# Copy application code
COPY . .

# Expose port
EXPOSE {self.config['system']['port']}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.config['system']['port']}/health || exit 1

# Start application
CMD ["python", "high_performance_hallucination_detector.py"]
"""
        
        with open(self.project_root / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        logger.info("✅ Dockerfile created")

    def _create_docker_compose(self):
        """Create docker-compose.yml"""
        logger.info("Creating docker-compose.yml...")
        
        compose_content = {
            'version': '3.8',
            'services': {
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': [f"{self.config['redis']['port']}:6379"],
                    'volumes': ['redis_data:/data'],
                    'restart': 'unless-stopped'
                },
                'hallucination-detector': {
                    'build': '.',
                    'ports': [f"{self.config['system']['port']}:{self.config['system']['port']}"],
                    'environment': {
                        'REDIS_HOST': 'redis',
                        'REDIS_PORT': '6379',
                        'REDIS_DB': '0'
                    },
                    'depends_on': ['redis'],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', f'http://localhost:{self.config["system"]["port"]}/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                }
            },
            'volumes': {
                'redis_data': None
            }
        }
        
        with open(self.project_root / "docker-compose.yml", 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
        
        logger.info("✅ docker-compose.yml created")

    def _run_docker_deployment(self):
        """Run Docker deployment"""
        logger.info("Building and running Docker containers...")
        
        try:
            # Build and start containers
            subprocess.run([
                "docker-compose", "up", "--build", "-d"
            ], check=True)
            
            logger.info("✅ Docker containers started")
            logger.info(f"🌐 Application available at: http://localhost:{self.config['system']['port']}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker deployment failed: {e}")
            raise

    def _deploy_aws(self):
        """Deploy to AWS"""
        logger.info("Deploying to AWS...")
        
        # Create AWS deployment files
        self._create_aws_files()
        
        # Deploy using AWS CLI or CloudFormation
        logger.info("Please deploy using AWS CLI or CloudFormation")
        logger.info("See aws_deployment/ directory for deployment files")

    def _deploy_gcp(self):
        """Deploy to Google Cloud Platform"""
        logger.info("Deploying to GCP...")
        
        # Create GCP deployment files
        self._create_gcp_files()
        
        # Deploy using gcloud CLI
        logger.info("Please deploy using gcloud CLI")
        logger.info("See gcp_deployment/ directory for deployment files")

    def _deploy_azure(self):
        """Deploy to Azure"""
        logger.info("Deploying to Azure...")
        
        # Create Azure deployment files
        self._create_azure_files()
        
        # Deploy using Azure CLI
        logger.info("Please deploy using Azure CLI")
        logger.info("See azure_deployment/ directory for deployment files")

    def _create_aws_files(self):
        """Create AWS deployment files"""
        aws_dir = self.project_root / "aws_deployment"
        aws_dir.mkdir(exist_ok=True)
        
        # Create CloudFormation template
        cloudformation_template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "High-Performance Hallucination Detection System",
            "Parameters": {
                "InstanceType": {
                    "Type": "String",
                    "Default": "t3.medium",
                    "Description": "EC2 instance type"
                }
            },
            "Resources": {
                "HallucinationDetectorEC2": {
                    "Type": "AWS::EC2::Instance",
                    "Properties": {
                        "InstanceType": {"Ref": "InstanceType"},
                        "ImageId": "ami-0c02fb55956c7d316",  # Amazon Linux 2
                        "SecurityGroups": [{"Ref": "SecurityGroup"}],
                        "UserData": {
                            "Fn::Base64": {
                                "Fn::Sub": |
                                    #!/bin/bash
                                    yum update -y
                                    yum install -y python3 pip redis
                                    systemctl start redis
                                    systemctl enable redis
                                    pip3 install -r requirements_high_performance.txt
                                    python3 high_performance_hallucination_detector.py
                            }
                        }
                    }
                }
            }
        }
        
        with open(aws_dir / "cloudformation.yaml", 'w') as f:
            yaml.dump(cloudformation_template, f, default_flow_style=False)

    def _create_gcp_files(self):
        """Create GCP deployment files"""
        gcp_dir = self.project_root / "gcp_deployment"
        gcp_dir.mkdir(exist_ok=True)
        
        # Create app.yaml for App Engine
        app_yaml = {
            "runtime": "python311",
            "entrypoint": "python high_performance_hallucination_detector.py",
            "env_variables": {
                "REDIS_HOST": "localhost",
                "REDIS_PORT": "6379"
            },
            "automatic_scaling": {
                "target_cpu_utilization": 0.6,
                "min_instances": 1,
                "max_instances": 10
            }
        }
        
        with open(gcp_dir / "app.yaml", 'w') as f:
            yaml.dump(app_yaml, f, default_flow_style=False)

    def _create_azure_files(self):
        """Create Azure deployment files"""
        azure_dir = self.project_root / "azure_deployment"
        azure_dir.mkdir(exist_ok=True)
        
        # Create Azure Container Instances deployment
        aci_deployment = {
            "apiVersion": "2018-10-01",
            "location": "[resourceGroup().location]",
            "name": "hallucination-detector",
            "properties": {
                "containers": [
                    {
                        "name": "hallucination-detector",
                        "properties": {
                            "image": "hallucination-detector:latest",
                            "ports": [
                                {
                                    "port": self.config['system']['port']
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": 1.0,
                                    "memoryInGB": 2.0
                                }
                            }
                        }
                    }
                ],
                "osType": "Linux",
                "ipAddress": {
                    "type": "Public",
                    "ports": [
                        {
                            "protocol": "tcp",
                            "port": self.config['system']['port']
                        }
                    ]
                }
            }
        }
        
        with open(azure_dir / "aci-deployment.json", 'w') as f:
            json.dump(aci_deployment, f, indent=2)

    def stop(self):
        """Stop the deployment"""
        logger.info("Stopping deployment...")
        
        try:
            # Stop Docker containers if running
            subprocess.run(["docker-compose", "down"], check=True)
            logger.info("✅ Docker containers stopped")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Stop Redis
        try:
            subprocess.run(["redis-cli", "shutdown"], check=True)
            logger.info("✅ Redis stopped")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def status(self):
        """Check deployment status"""
        logger.info("Checking deployment status...")
        
        # Check if application is running
        try:
            import requests
            response = requests.get(f"http://localhost:{self.config['system']['port']}/health")
            if response.status_code == 200:
                logger.info("✅ Application is running")
                health_data = response.json()
                logger.info(f"   Status: {health_data.get('status', 'unknown')}")
                logger.info(f"   Uptime: {health_data.get('uptime', 0):.2f} seconds")
            else:
                logger.warning("⚠️ Application is not responding")
        except Exception as e:
            logger.warning(f"⚠️ Cannot connect to application: {e}")
        
        # Check Redis
        try:
            import redis
            r = redis.Redis(host=self.config['redis']['host'], port=self.config['redis']['port'])
            r.ping()
            logger.info("✅ Redis is running")
        except Exception as e:
            logger.warning(f"⚠️ Redis is not available: {e}")

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy High-Performance Hallucination Detection System")
    parser.add_argument("command", choices=["deploy", "stop", "status"], help="Deployment command")
    parser.add_argument("--type", choices=["local", "docker", "aws", "gcp", "azure"], 
                       default="local", help="Deployment type")
    parser.add_argument("--config", default="deployment_config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    deployer = HighPerformanceDeployer(args.config)
    
    try:
        if args.command == "deploy":
            if args.type == "local":
                deployer.deploy_local()
            elif args.type == "docker":
                deployer.deploy_docker()
            elif args.type in ["aws", "gcp", "azure"]:
                deployer.deploy_cloud(args.type)
        elif args.command == "stop":
            deployer.stop()
        elif args.command == "status":
            deployer.status()
    
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 