import os
try:

    from setuptools import setup, find_packages
except ImportError:
    raise RuntimeError('setuptools is required')


import versioneer


PACKAGE = 'multi_loc'


SHORT_DESC = 'Tools for localization on multiple scales'
AUTHOR = 'Travis Harty'

setup(
    name=PACKAGE,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=SHORT_DESC,
    author=AUTHOR,
    packages=find_packages(),
    include_package_data=True,
    scripts=[os.path.join('scripts', s) for s in os.listdir('scripts')]
)
