from setuptools import setup, find_packages

setup(
    name="open_trust_eval",
    version="0.1.0",
    description="OpenTrustEval (OTE) - A Universal Trustworthy AI Framework",
    author="xAI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers==4.41.1",
        "torch==2.3.0",
        "qiskit==1.0.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "tensorflow==2.16.1",
        "datasets==2.19.0"
    ],
    python_requires=">=3.8",
)
