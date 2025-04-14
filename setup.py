from setuptools import find_packages, setup

setup(
    name="bitsandbytes-intel",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["bitsandbytes"],
    entry_points={"torch.backends": ["bitsandbytes_intel = bitsandbytes_intel:_autoload"]},
)
