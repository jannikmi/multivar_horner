# -*- coding:utf-8 -*-
import os
import re

from setuptools import setup


def get_version(package):
    """
    Return package version as listed in `__version__` in `__init__.py`.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


version = get_version('multivar_horner')

setup(
    name='multivar_horner',
    version=version,
    packages=['multivar_horner'],
    description='python package implementing a multivariate Horner scheme for efficiently '
                'evaluating multivariate polynomials',
    author='J. Michelfeit',
    author_email='python@michelfe.it',
    license='MIT licence',
    url='https://github.com/MrMinimal64/multivar_horner',  # use the URL to the github repo
    keywords='math mathematics polynomial polynomials polynomial-evaluation multivariate multivariate-polynomials'
             ' horner horner-scheme factorisation factorization',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    long_description='Python package for computing Horner factorisations of multivariate polynomials '
                     'for efficient evaluation.\n'
                     'Please check GitHub for the documentation with plots: \n'
                     'https://github.com/MrMinimal64/multivar-horner',
    install_requires=[
        'numpy',
        'numba',
    ],
)
