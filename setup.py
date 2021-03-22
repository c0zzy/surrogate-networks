from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='surrogate_networks',
    version='1.0',
    packages=find_packages(),  #same as name
    install_requires=required, #external packages as dependencies
)