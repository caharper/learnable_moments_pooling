from setuptools import setup, find_packages
from setuptools.dist import Distribution
import pathlib


# TODO: ADD LISCENCE

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


setup(
    name="package-name",
    version="0.1",
    long_description=README,
    long_description_content_type="text/markdown",
    url="enter_github_url",
    author="author_name",
    author_email="author_email",
    install_requires=[],
    extras_require={},
    python_requires=">=3.7",
    distclass=Distribution,
    packages=find_packages(exclude=("*_test.py",)),
    include_package_data=True,
)
