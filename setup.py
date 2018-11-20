"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('src/pyccutof/version.py').read())

setup(
    name='pyccutof',
    version=__version__,
    description='Work with mass spectrometer data files',
    long_description=long_description,
    url='https://github.com/awbirdsall/pyccutof',
    author='Adam Birdsall',
    author_email='abirdsall@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=['mass spectometry', 'chemistry'],
    package_dir = {'': 'src'},
    packages=['pyccutof'],
    install_requires=['matplotlib>=1.5','numpy','pandas','scipy>=1.1'],
    # avoid trying (and failing) to install pyodbc in readthedocs
    on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
    if not on_rtd:
        install_requires.append('pyodbc')
    package_data={
        'pyccutof': ['data/sample.FFC', 'data/sample.FFT']
    },
    include_package_data=True,
    entry_points={
        # entry point for command line interface would go here
        # 'console_scripts': [
        #     'pyccutof=pyccutof.command_line:command',
        # ],
    },
)
