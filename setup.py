from setuptools import setup, find_packages

setup(
    name="wordle",
    version="0.1.0",
    packages=find_packages(include=["wordle", "wordle.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2",
    ],
)
