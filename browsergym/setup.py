# deprecated
from setuptools import find_packages, setup
from io import open


def read_requirements_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]


setup(
    name="tailored_browsergym",
    version="0.0.1",
    author="Yitao Liu",
    author_email="lyitao17@gmail.com",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    package_dir={'': 'src'},
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={},
    include_package_data=True,
    python_requires=">=3.6",
    tests_require=["pytest"],
)