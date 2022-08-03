import init
import setuptools

long_description = """
The independent util library for TensorNeko. This library doesn't require PyTorch as a dependency.
"""

requirements = []
with open("requirements_util.txt", "r") as file:
    for line in file:
        requirements.append(line.strip())

version = init.read_version()
init.write_version(version)

setuptools.setup(
    name="tensorneko_util",
    version=version,
    author="ControlNet",
    author_email="smczx@hotmail.com",
    description="The Utils for Library TensorNeko.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ControlNet/tensorneko",
    project_urls={
        "Bug Tracker": "https://github.com/ControlNet/tensorneko/issues",
        "Source Code": "https://github.com/ControlNet/tensorneko",
    },
    keywords=["deep learning", "pytorch", "AI"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["tensorneko_util", "tensorneko_util.*"]),
    package_data={
        "tensorneko_util": [
            "version.txt",
            "visualization/watcher/web/dist/index.html",
            "visualization/watcher/web/dist/assets/*",
        ]
    },
    python_requires=">=3.7",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities"
    ],
)
