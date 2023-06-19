from setuptools import setup, find_packages

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='g4satbench',
    version='1.0.0',
    description='A Comprehensive Benchmark on GNNs for SAT Solving',
    packages=find_packages(),
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)