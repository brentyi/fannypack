from setuptools import setup

setup(
    name="hfdsajk",
    version="0.0",
    description="PyTorch utilities",
    url="http://github.com/brentyi/hfdsajk",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=["hfdsajk"],
    install_requires=[
        'h5py',
        'torch',
        'numpy',
        'matplotlib',
        'tensorboard',
        'jupyter',
        'scipy',
        'tqdm'
    ],
)
