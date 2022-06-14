#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup

setup(
    name="multivar_horner",
    packages=["multivar_horner"],
    description="python package implementing a multivariate Horner scheme for efficiently "
    "evaluating multivariate polynomials",
    long_description="python package implementing a multivariate Horner scheme "
    "for efficiently evaluating multivariate polynomials \n\n"
    "for more refer to the documentation: https://multivar_horner.readthedocs.io/en/latest/",
    # version: in VERSION file https://packaging.python.org/guides/single-sourcing-package-version/
    # With this approach you must make sure that the VERSION file is included in all your source
    # and binary distributions (e.g. add include VERSION to your MANIFEST.in).
    author="Jannik Michelfeit",
    author_email="python@michelfe.it",
    license="MIT licence",
    url="https://github.com/jannikmi/multivar_horner",  # use the URL to the github repo
    project_urls={
        "Source Code": "https://github.com/jannikmi/multivar_horner",
        "Documentation": "https://multivar_horner.readthedocs.io/en/latest/",
        "Changelog": "https://github.com/jannikmi/multivar_horner/blob/master/CHANGELOG.rst",
    },
    keywords="math mathematics polynomial polynomials polynomial-evaluation multivariate multivariate-polynomials"
    " horner horner-scheme factorisation factorization",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={"numba": ["numba>=0.55"]},
    python_requires=">=3.7",
)
