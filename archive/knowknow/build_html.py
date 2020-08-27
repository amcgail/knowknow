# ipython nbconvert --config mycfg.py
import sys; sys.path.append('..')
from knowknow import *

c = get_config()
nbs = list(str(x) for x in Path(BASEDIR).glob("**/*.ipynb"))
nbs = [x for x in nbs if '.ipynb_checkpoints' not in x]

c.NbConvertApp.notebooks = nbs
#c.TemplateExporter.exclude_input = True # gets rid of Code input
c.NbConvertApp.export_format = 'html'
c.FilesWriter.relpath = BASEDIR
c.FilesWriter.build_directory = ''#"./html"