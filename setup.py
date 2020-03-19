from setuptools import setup

setup(
    name="fannypack",
    version="0.0",
    description="PyTorch utilities",
    url="http://github.com/brentyi/fannypack",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=["fannypack"],
    install_requires=[
        "dill",
        "h5py",
        "numpy",
        "pyyaml",
        "tensorboard",
        "torch",
    ],
)
