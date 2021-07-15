import setuptools
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()
    
class InstallWrapper(install):
    """Provides a install wrapper to handle set-up of dependencies."""
    def run(self):
        install.run(self)
        self.setup_nltk()

    def setup_nltk(self):
        import nltk
        nltk.download('stopwords')

setuptools.setup(
    name="knowknow-amcgail", # Replace with your own username
    version="0.3.1",
    author="Alec McGail",
    author_email="amcgail2@gmail.com",
    description="Analyzing the evolution of ideas using citation analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amcgail/knowknow",
    packages=setuptools.find_packages(),
    package_data={
        "knowknow":[
            #"analyses/*",
            #"creating variables/*",
            #"external-data/*",
            #"utilities/*",
            #"writeups/*",
            "*.*"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "colour>=0.1.5",
        "cycler>=0.10.0",
        "editdistance>=0.5.3",
        "ipython>=7.14.0",
        "jupyter>=1.0.0",
        "IPython",
        "jupyterlab>=2.1.2",
        "lxml>=4.4.1",
        "matplotlib>=3.2.1",
        "nameparser>=1.0.6",
        "networkx>=2.3",
        "nltk>=3.5",
        "numpy>=1.18.4",
        "pandas>=1.0.3",
        "PyYAML>=5.3.1",
        "scikit-learn>=0.23.0",
        "scipy>=1.4.1",
        "seaborn>=0.10.1",
        "six>=1.14.0",
        "statsmodels>=0.11.1",
        "string-grouper>=0.1.0",
        "tabulate>=0.8.7",
    ],
    cmdclass={'install': InstallWrapper},
)