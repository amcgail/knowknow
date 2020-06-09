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
    version="0.1.3",
    author="Alec McGail",
    author_email="amcgail2@gmail.com",
    description="Analyzing the evolution of ideas using citation analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amcgail/knowknow",
    packages=setuptools.find_packages(),
    package_data={
        "":["**/*.py", "**/*.yaml", "**/*.ipynb", "writeups","external-data"],
        "knowknow":[]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    cmdclass={'install': InstallWrapper},
)