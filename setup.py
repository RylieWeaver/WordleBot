from setuptools import setup, find_packages

setup(
    name="wordlebot",               # pip install wordlebot-0.1.0.whl
    version="0.1.0",
    packages=find_packages(include=["wordle*"]),  # ONLY install wordle/*
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2",               # match the cu118 wheel as discussed
    ],
)
