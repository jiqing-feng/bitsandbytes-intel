from setuptools import setup, find_packages

setup(
    name="bnb_intel",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["bitsandbytes"],
    entry_points={"torch.backends": ["bnb_intel = bnb_intel:_autoload"]},
)
