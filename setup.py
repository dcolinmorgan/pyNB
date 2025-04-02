from setuptools import setup, find_packages

setup(
    name="pyNB",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "pandas",
        "matplotlib",
        "requests",
    ],
    author="D.C.Morgan",
    description="A Python port of the GENESpider MATLAB package",
    python_requires=">=3.9",
)
