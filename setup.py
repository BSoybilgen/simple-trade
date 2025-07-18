import os
from setuptools import setup, find_packages

setup(
    name="simple_trade",
    description="Compute technical indicators and build trade strategies in a simple way",
    license='AGPL-3.0',
    version="0.1.3",
    author="Baris Soybilgen",
    packages=find_packages(),
    project_urls={
        'Source': 'https://github.com/BSoybilgen/simple-trade/',
    },
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md'),
                              encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "yfinance>=0.2.59",
        "pandas>=1.4.0",
        "numpy>=1.22.0",
        "joblib>=1.1.1",
        "matplotlib>=3.6.0"
    ],
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Visualization',
        ],
    keywords=[
            'algo',
            'algorithmic',
            'backtest',
            'backtesting',
            'candlestick',
            'chart',
            'crypto',
            'currency',
            'equity',
            'exchange',
            'finance',
            'financial',
            'forex',
            'fx',
            'indicator',
            'invest',
            'investing',
            'investment',
            'ohlc',
            'ohlcv',
            'order',
            'price',
            'profit',
            'quant',
            'quantitative',
            'simulation',
            'stocks',
            'strategy',
            'ticker',
            'trader',
            'trading',
        ],
)
