from setuptools import setup, find_packages

setup(
    name="psychological-state-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
)
