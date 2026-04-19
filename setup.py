import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensor_talezz_rv",
    version="0.2.0",
    author="RV Patil",
    description="Making AI Approachable, Interpretable, and Resource-Friendly.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rvpatil-tech/TensorTalezz-RV",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "joblib>=1.1.0"
    ],
)
