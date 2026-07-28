"""Package setup for icecream-shelf-detector."""
from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
long_description = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="icecream-shelf-detector",
    version="1.0.0",
    description="AI-powered vending machine ice cream shelf analyzer using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Internship Project",
    python_requires=">=3.10",
    package_dir={"": "."},
    packages=find_packages(where="."),
    install_requires=[
        "ultralytics>=8.2",
        "torch>=2.3",
        "opencv-python>=4.9",
        "albumentations>=1.4",
        "streamlit>=1.35",
        "plotly>=5.22",
        "pandas>=2.2",
        "numpy>=1.26",
        "Pillow>=10.3",
        "pyyaml>=6.0",
        "requests>=2.32",
        "beautifulsoup4>=4.12",
        "click>=8.1",
        "scikit-learn>=1.4",
        "tqdm>=4.66",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "jupyter", "ipykernel"],
    },
    entry_points={
        "console_scripts": [
            "shelf-train=src.models.trainer:main",
            "shelf-eval=src.models.evaluator:main",
            "shelf-predict=src.models.predictor:main",
            "shelf-validate=src.data.validator:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
