import os
import re
import setuptools

NAME = "discopula"
AUTHOR = "Dhyey Mavani"
AUTHOR_EMAIL = "ddmavani2003@gmail.com"
DESCRIPTION = "This package is for discrete copula modeling and implementation of new scoring methods pertaining to ordinal and categorical discrete data."
LICENSE = "MIT"
KEYWORDS = "discrete-copula"
URL = "https://github.com/dmavani25/" + NAME
README = ".github/README.md"
CLASSIFIERS = [
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
INSTALL_REQUIRES = ["numpy", "scipy", "matplotlib"]
ENTRY_POINTS = {}
SCRIPTS = []

HERE = os.path.dirname(__file__)


def read(file):
    with open(os.path.join(HERE, file), "r") as fh:
        return fh.read()


VERSION = re.search(
    r'__version__ = [\'"]([^\'"]*)[\'"]', read(NAME.replace("-", "_") + "/__init__.py")
).group(1)

LONG_DESCRIPTION = read(README)

if __name__ == "__main__":
    setuptools.setup(
        name=NAME,
        version=VERSION,
        packages=setuptools.find_packages(),
        author=AUTHOR,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license=LICENSE,
        keywords=KEYWORDS,
        url=URL,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        scripts=SCRIPTS,
        include_package_data=True,
        python_requires=">=3.8",
    )
