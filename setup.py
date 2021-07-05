import setuptools

with open("README.md", "r") as file:
    long_description = file.read()


requirements = []
with open("requirements.txt", "r") as file:
    for line in file:
        requirements.append(line.strip())


with open("version.txt", "r") as file:
    version = file.read()

setuptools.setup(
    name="tensorneko",
    version=version,
    author="ControlNet",
    author_email="smczx@hotmail.com",
    description="A small package for PyTorch utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ControlNet/tensorneko",
    project_urls={
        "Bug Tracker": "https://github.com/ControlNet/tensorneko/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities"
    ],
)
