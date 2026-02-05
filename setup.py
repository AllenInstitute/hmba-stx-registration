from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="hmba_registration",
    version="0.1",
    description="Tools for registration of HMBA data to Allen Brain Atlas",
    author="Mike Huang",
    author_email="mike.huang@alleninstitute.org",
    url="https://github.com/AllenInstitute/hmba_registration",
    license=license,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
)
