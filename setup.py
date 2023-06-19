from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='g4satbench',
    version='1.0.0',
    description='g4satbench',
    packages=['g4satbench'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)