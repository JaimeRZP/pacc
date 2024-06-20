#!/usr/bin/python3

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(
    name='pacc',
    version='0.1.0',
    description='plot all the correlations and covariances',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jaimerzp/pacc',
    author='Jaime Ruiz Zapatero',
    author_email='jaime.ruiz-zapatero@ucl.ac.uk',
    license="GPLv2",
    packages=['pacc'],
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Physics"
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
