from setuptools import setup, find_packages

setup(
    name='ptycho',
    version='0.1.0',
    packages=find_packages(),  # Finds all packages under the project
    install_requires=[],       # Add any external dependencies here (e.g., numpy)
    author='Efe Tarhan',
    author_email='efe.tarhan@epfl.ch',
    description='A Python library for ptychography and related algorithms',
    #long_description=open('README.md').read(),  # Include a README if you have one
    long_description_content_type='text/markdown',
    url='https://github.com/tarhanefe/epfl_semester_project/ptycho',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python version required
)