from setuptools import setup, find_packages
from codecs import open
from os import path


with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup_info = dict(
     name='factual_evaluation_source_based',
     version='1.0.0',
     author='Sanjana Ramprasad',
     author_email='ramprasad.sa@northeastern.edu',
     url='https://github.com/sanjanaramprasad/factual_evaluation_source_based.git',
     description='Factual evaluation dialogue summaries',
     long_description=long_description,
     packages=find_packages()
)

setup(**setup_info)