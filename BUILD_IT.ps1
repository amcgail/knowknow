# to build the python package and upload it
Remove-Item dist/*
python setup.py sdist bdist_wheel
twine upload dist/*
# pip install --upgrade knowknow-amcgail

# to build the HTML files
# jupyter nbconvert --config build_html.py