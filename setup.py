
import sys
from setuptools import setup, find_packages
import os
from setuptools import find_packages
import setuptools
from setuptools.command.install_scripts import install_scripts
from setuptools.command.install import install
from distutils import log

if sys.version_info.major < 3:
    sys.exit('Sorry, sierra-local requires Python 3.x')

def _requires_from_file(filename):
    print(os.getcwd())
    return open(filename).read().splitlines()

if __name__ == "__main__":
                
    setup(
        name='Attention-PHV',
        version='0.0.1',
        description="This package called Attention-PHV is used for protein-protein interaction (PPI) prediction",
        author="Kyushu institute of technology. Kurata laboratory.",
        install_requires = _requires_from_file('requirements.txt'),
        packages = ["."],
        entry_points={
            'console_scripts':[
                'aphv = main:main',
            ]
        },
        classifiers=[
        "Programming Language :: Python :: 3.8.0",
        "License :: Apache License 2.0"
        ],
    )































