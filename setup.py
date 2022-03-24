# encoding: utf-8

"""
Read R datasets from Python.

This package parses .rda datasets used in R. It does not depend on the R
language or its libraries, and thus it is released under a MIT license.
"""
import os
import pathlib
import sys

from setuptools import find_packages, setup

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

DOCLINES = (__doc__ or '').split("\n")

with open(
    pathlib.Path(os.path.dirname(__file__)) / 'rdata' / 'VERSION',
    'r',
) as version_file:
    version = version_file.read().strip()

setup(
    name='rdata',
    version=version,
    description=DOCLINES[1],
    long_description="\n".join(DOCLINES[3:]),
    url='https://github.com/vnmabus/rdata',
    author='Carlos Ramos Carreño',
    author_email='vnmabus@gmail.com',
    include_package_data=True,
    platforms=['any'],
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.7, <4',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Typing :: Typed',
    ],
    keywords=['rdata', 'r', 'dataset'],
    install_requires=[
        'numpy',
        'xarray',
        'pandas',
    ],
    setup_requires=pytest_runner,
    tests_require=[
        'pytest-cov',
        'numpy>=1.14',  # The printing format for numpy changes
    ],
    test_suite='rdata.tests',
    zip_safe=False,
)
