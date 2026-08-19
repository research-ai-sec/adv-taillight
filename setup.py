import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vanishing-vehicles",
    version="1.0.0",
    description=(
        "Physical Adversarial Taillights Against 3D Occupancy Networks "
        "(GAN fine-tuning attack on SurroundOcc)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    license="CC-BY-NC-4.0",
    packages=find_packages(exclude=("SurroundOcc", "data")),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "torchvision>=0.11",
        "numpy>=1.20",
        "scipy>=1.7",
        "pandas>=1.3",
        "PyYAML>=5.4",
        "opencv-python>=4.5",
        "Pillow>=8.4",
        "matplotlib>=3.4",
        "imageio>=2.9",
        "click>=8.0",
        "tqdm>=4.62",
        "requests>=2.26",
        "dill>=0.3.4",
        "loguru>=0.7",
        "timm>=0.9",
        "ftfy>=6.1",
        "regex>=2022.1.18",
        "pyspng>=0.1.0",
        "nuscenes-devkit>=1.1",
        "pyquaternion>=0.9.9",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
