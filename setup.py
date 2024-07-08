from setuptools import setup, find_packages

setup(
    name="gaussian_noise_reduction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.3.56",
        "matplotlib>=3.4.2",
        "scikit-image>=0.18.2",
    ],
    entry_points={
        "console_scripts": [
            "reduce_noise=src.main:main",
        ],
    },
)
