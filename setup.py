from setuptools import setup, find_packages

setup(
    name="simple_trading",
    version="0.1.0",
    description="Download financial data and compute technical indicators.",
    author="Your Name", # Consider changing this
    packages=find_packages(),
    install_requires=[
        "yfinance",
        "pandas",
        "numpy"
    ],
    python_requires='>=3.7',
)
