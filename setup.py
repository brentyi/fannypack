from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fannypack",
    version="0.0.3-pre",
    description="PyTorch utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/fannypack",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(exclude=["examples", "tests"]),
    package_data={"fannypack": ["py.typed"]},
    python_requires=">=3.6",
    install_requires=["dill", "h5py", "numpy", "pyyaml", "tensorboard", "torch",],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
