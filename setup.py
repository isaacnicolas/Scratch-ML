from setuptools import setup, find_packages

setup(
    name='scratchml',
    version='0.1',
    author='Isaac Nicolas',
    author_email='isaacnicolas97@gmail.com',
    description='A custom machine learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/isaacnicolas/scratchml',
    packages=find_packages(),
    install_requires=['numpy','scikit-learn'],
    license='MIT'
)
