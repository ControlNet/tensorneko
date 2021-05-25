#!/bin/sh

rm -rf dist/
rm -rf build/
rm -rf src/tensorneko.egg-info/

python setup.py sdist bdist_wheel
twine upload dist/*