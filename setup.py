import os
from setuptools import setup, find_packages

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = open(os.path.join(PACKAGE_ROOT, "README.md"), "r").read()

# TODO Update install requirements to add transformers, zarr

if __name__ == "__main__":
    setup(
        name="rootflow-datasets",
        version="0.0.0",
        description="Base classes, tools and utilities for rootflow datasets. ",
        long_description=README_FILE,
        long_description_content_type="text/markdown",
        url="https://github.com/with-branch/rootflow-datasets",
        author="GenerallyIntelligent",
        author_email="tannersims@generallyintelligent.me",
        license="MIT",
        packages=find_packages(),
        install_requires=[
            "torch >=1.10.0, <2.0.0",
        ],
    )
