import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="knowknow-amcgail", # Replace with your own username
    version="0.0.1",
    author="Alec McGail",
    author_email="amcgail2@gmail.com",
    description="Analyzing the evolution of ideas using citation analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amcgail/knowknow",
    packages=setuptools.find_packages(),
    package_data={
        "science2":["variables/*","figures/*","creating variables/*"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)