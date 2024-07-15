import init
import setuptools

long_description = """
The CLI tools developed based on TensorNeko. This library use tensorneko_util as the dependency.
"""

requirements = []
with open("requirements_tool.txt", "r") as file:
    for line in file:
        requirements.append(line.strip())

version = init.read_version()
init.write_version(version)
requirements.append(f"tensorneko_util == {version}")

setuptools.setup(
    name="tensorneko_tool",
    version=version,
    author="ControlNet",
    author_email="smczx@hotmail.com",
    description="The CLI Tools for Library TensorNeko.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ControlNet/tensorneko",
    project_urls={
        "Bug Tracker": "https://github.com/ControlNet/tensorneko/issues",
        "Source Code": "https://github.com/ControlNet/tensorneko",
    },
    keywords=["deep learning", "pytorch", "AI", "data processing"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["tensorneko_tool", "tensorneko_tool.*"]),
    package_data={
        "tensorneko_tool": [
            "version.txt",
        ]
    },
    # cli tools
    entry_points={
        "console_scripts": [
            "tensorneko = tensorneko_tool.__main__:main",
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
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities"
    ],
)
