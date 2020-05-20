# ipython nbconvert --config mycfg.py
from knowknow import BASEDIR, Path

c = get_config()
c.NbConvertApp.notebooks = list(str(x) for x in Path(BASEDIR).glob("**/*.ipynb"))
c.NbConvertApp.export_format = 'html'
c.FilesWriter.build_directory = '.'