import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())


setuptools.setup(
    name="tensorneko",
    version="0.0.0-3",
    author="ControlNet",
    author_email="smczx@hotmail.com",
    description="A small package for TensorFlow2 utils",
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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities"
    ],
)
