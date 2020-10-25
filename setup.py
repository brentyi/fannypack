from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fannypack",
    version="0.0.20",
    description="PyTorch utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/fannypack",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(exclude=["examples", "tests"]),
    package_data={"fannypack": ["py.typed"]},
    entry_points={"console_scripts": ["buddy = fannypack.scripts.buddy_cli:main"]},
    python_requires=">=3.7",
    install_requires=[
        "argcomplete",
        "beautifultable >= 1.0.0",
        "dill",
        "h5py",
        "numpy",
        # "numpy-stubs @ https://github.com/numpy/numpy-stubs/tarball/master",
        "pygments",
        "pyyaml",
        "simple_term_menu",
        "tensorboard",
        "termcolor",
        "torch",
        "tqdm",
        "Pillow",  # Needed for logging images in Tensorboard
        # "pip>=20.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
