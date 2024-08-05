import init
import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

requirements = []
extra_requirements = []
with open("requirements.txt", "r") as file:
    for line in file:
        if line.startswith("lightning") or line.startswith("tensorboard"):
            extra_requirements.append(line.strip())
        else:
            requirements.append(line.strip())


version = init.read_version()
init.write_version(version)
requirements.append(f"tensorneko_util == {version}")

setuptools.setup(
    name="tensorneko",
    version=version,
    author="ControlNet",
    author_email="smczx@hotmail.com",
    description="Tensor Neural Engine Kompanion. An util library based on PyTorch and PyTorch Lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ControlNet/tensorneko",
    project_urls={
        "Bug Tracker": "https://github.com/ControlNet/tensorneko/issues",
        "Source Code": "https://github.com/ControlNet/tensorneko",
    },
    keywords=["deep learning", "pytorch", "AI", "data processing"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["tensorneko", "tensorneko.*"]),
    package_data={
        "tensorneko": [
            "version.txt"
        ]
    },
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "lightning": extra_requirements
    },
    classifiers=[
        "Programming Language :: Python :: 3",
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
