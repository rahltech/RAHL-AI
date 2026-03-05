from setuptools import setup, find_packages

setup(
    name="rahl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RAHL - Realistic AI for High-quality video generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rahltech/RAHL-AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
)
