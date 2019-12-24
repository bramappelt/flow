import os
import setuptools


with open("README.md", "r") as fr:
    long_description = fr.read()

# link requirements.txt
requirements = []
if os.path.isfile("requirements.txt"):
    with open("requirements.txt", "r") as fr:
        requirements = fr.read().splitlines()

setuptools.setup(
    name="Flow-BramAppelt",
    version="0.0.4",
    author="Bram Berendsen",
    author_email="bram.berendsen@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bramappelt/flow",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)
