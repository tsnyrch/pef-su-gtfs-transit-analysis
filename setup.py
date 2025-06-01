from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="pef-su-gtfs-transit-analysis",
    version="1.0.0",
    author="Tomáš Snyrch, Martin Fiša",
    author_email="xsnyrch@mendelu.cz, xfisa@mendelu.cz",
    description="A comprehensive machine learning and network analysis platform for IDS JMK public transit data (Brno, Czech Republic)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsnyrch/pef-su-gtfs-transit-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/tsnyrch/pef-su-gtfs-transit-analysis/issues",
        "Documentation": "https://github.com/tsnyrch/pef-su-gtfs-transit-analysis/wiki",
        "Source Code": "https://github.com/tsnyrch/pef-su-gtfs-transit-analysis",
        "Data Source": "https://data.brno.cz/datasets/379d2e9a7907460c8ca7fda1f3e84328/about",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Transportation",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gtfs-analysis=main:main",
            "pef-su-gtfs=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "gtfs",
        "transit",
        "transportation",
        "machine learning",
        "network analysis",
        "data science",
        "public transport",
        "routing",
        "prediction",
        "visualization",
        "brno",
        "czech republic",
        "ids jmk",
        "mendelu",
        "pef",
    ],
    zip_safe=False,
)
