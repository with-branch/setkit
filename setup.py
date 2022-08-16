import os
from setuptools import setup, find_packages

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = open(os.path.join(PACKAGE_ROOT, "README.md"), "r").read()

# TODO Update install requirements to add transformers, zarr

if __name__ == "__main__":
    setup(
        name="setkit",
        version="0.0.0",
        description="Smooth and easy ways to create highly functional datasets",
        long_description=README_FILE,
        long_description_content_type="text/markdown",
        url="https://github.com/with-branch/setkit",
        author="HesitantlyHuman",
        author_email="tannersims@hesitantlyhuman.com",
        license="MIT",
        packages=find_packages(),
        install_requires=[],
    )
