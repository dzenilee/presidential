from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    # Application name:
    name="presidential",
    # Version number (initial):
    version="0.0.1",
    author="Jenny Lee",
    author_email="dzenilee@gmail.com",
    description="presidential package",
    long_description=long_description,
    url="https://github.com/dzenilee/presidential",
    # Include the packages, exclude tests
    packages=find_packages("src", exclude=["test_*.py", "*_test.py"]),
    include_package_data=True,
    package_dir={"": "src"},
    # license="LICENSE.txt",
)
