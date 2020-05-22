# ipython nbconvert --config mycfg.py
from knowknow import BASEDIR, Path

c = get_config()
nbs = list(str(x) for x in Path(BASEDIR).glob("analyses/**/*.ipynb"))
nbs = [x for x in nbs if '.ipynb_checkpoints' not in x]
nbs.append(str(Path(BASEDIR).joinpath("all_figures.ipynb")))

c.NbConvertApp.notebooks = nbs
c.NbConvertApp.export_format = 'html'
c.FilesWriter.build_directory = str(Path(__file__).parent.joinpath("knowknow_html"))