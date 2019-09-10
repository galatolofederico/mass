import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mass",
    version="0.1.0",
    author="Federico A. Galatolo, Diego Casu, Filippo Minutella",
    author_email="federico.galatolo@ing.unipi.it",
    description="mass: Many Agent Stigmergy Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/galatolofederico/mass",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pyqtgraph",
        "Pillow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta"
    ],
)