

__all__ = [
    'figure_dir',
    'GLOBS','cfile'
]

import os
from pathlib import Path
import yaml

#print(os.environ)

# more local stuff
if  'NBDIR' in os.environ:
    notebook_dir = Path(os.environ['NBDIR']).parent
    figure_dir = notebook_dir.joinpath('figures')
    #table_dir = notebook_dir.joinpath('tables')


    for d in [figure_dir]:#,table_dir]:
        if not d.exists():
            print("Creating directory: %s" % d)
            d.mkdir()

cfile = Path(__file__).parent.joinpath('config.yaml')

if cfile.exists():
    with cfile.open('r') as f:
        GLOBS = yaml.load(f, Loader=yaml.FullLoader)
else:
    GLOBS = {}

if 'kkdir' in GLOBS:
    #variable_dir = Path(os.environ['VARDIR']) if 'VARDIR' in os.environ else None
    variable_dir = Path(GLOBS['kkdir'], 'data')