__all__ = [
    'variable_dir',
    'figure_dir',
    'modules'
]

import os
from pathlib import Path

variable_dir = Path(os.environ['VARDIR'])

# more local stuff
notebook_dir = Path(os.environ['NBDIR']).parent
figure_dir = notebook_dir.joinpath('figures')
table_dir = notebook_dir.joinpath('tables')


for d in [figure_dir,table_dir,variable_dir]:
    if not d.exists():
        print("Creating directory: %s" % d)
        d.mkdir()