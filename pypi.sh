#!/bin/sh

# build and upload tensorneko_util
rm -rf dist/
rm -rf build/
rm -rf src/tensorneko_util.egg-info/
python setup_util.py sdist bdist_wheel
twine upload dist/*

# build and upload tensorneko_tool
rm -rf dist/
rm -rf build/
rm -rf src/tensorneko_tool.egg-info/
python setup_tool.py sdist bdist_wheel
twine upload dist/*

# build and upload tensorneko
rm -rf dist/
rm -rf build/
rm -rf src/tensorneko.egg-info/
python setup.py sdist bdist_wheel
twine upload dist/*