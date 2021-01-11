from . import env, Path
import papermill as pm

__all__ = [
    'MetaNB'
]

class MetaNB:

    def __init__(self, name, debug=False):
        self.full_name = name
        self.space, self.short_name = self.full_name.split("/")
        if self.space == '.':
            self.spacefn = env.notebook_dir
        else:
            self.spacefn = env.modules[self.space]
        self.fn = Path(self.spacefn, self.short_name + ".ipynb")

        assert(self.fn.exists())
    
    def __call__(self, **params):
        
        pm.execute_notebook(
            str(self.fn),
            str(self.fn),
            parameters = params,
            nest_asyncio=True
        )