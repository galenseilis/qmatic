rm -r dist/

python -m build

pip install --force-reinstall dist/qmatic-0.0.0-py3-none-any.whl
