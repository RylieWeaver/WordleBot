from setuptools import setup, find_packages

setup(
    name="wordle",
    version="0.1.1",
    packages=find_packages(include=["wordle", "wordle.*"]),
    include_package_data=True,
    package_data={
        "wordle": ["**/*.json", "**/*.txt"],
    },
    python_requires=">=3.9",
    install_requires=["torch>=2.2"],
)
