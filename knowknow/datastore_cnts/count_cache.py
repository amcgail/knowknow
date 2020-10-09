from .. import env, defaultdict, pd

import pickle
from .. import Path, Counter, defaultdict, download_file, VariableNotFound

__all__ = [
    'Dataset'
]

if False:
    import sys
    #sys.path.append('C:/Users/amcga/GitHub/knowknow/archive')
    sys.path.append('C:/Users/amcga/GitHub/knowknow')
    sys.path = sys.path[1:]
    print(sys.path)
    from knowknow.datastore_cnts.count_cache import *
    wos = Dataset('sociology-wos')
    wos(fy=1993)


class CitedRef:
    def __init__(self, identifier):
        self.identifier = identifier

        isplit = self.identifier.split("|")

        self.full_ref = self.identifier
        self.author = isplit[0]
        self.pyear = None
        if len(isplit) == 2:
            self.type = 'book'
        else:
            self.type = 'article'
            self.pyear = int(isplit[1])        

    def __repr__(self):
        return "( %s )" % "Cited %s: %s" % (self.type, self.full_ref)

class ent:
    citing_author = 'fa'
    cited_author = 'ta'
    publication_year = 'fy'
    cited_year = 'ty'
    cited_pub = 'c'

    @classmethod
    def c_fn(cls, x):
        return CitedRef(x)

    ents = 'fa,ta,fy,ty,c'.split(",")



class Constants:

    DEFAULT_KEYS = ['fj.fy', 'fy']  # , 'c', 'c.fy']
    data_files = {
        'sociology-wos': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/'
                         '5eded795c67d30014e1f3714/?zip=',
        'sociology-jstor': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/'
                           '5eded76fc67d3001491f27d1/?zip=',
        'sociology-jstor-basicall': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/'
                                   '5eded7478b542601748b4bdb/?zip='
    }


save_nametuples = {}
def make_cross(*args, **kwargs):
    global save_nametuples
    from collections import namedtuple
    
    if len(args):
        assert(type(args[0]) == dict)
        return make_cross(**args[0])
    
    keys = tuple(sorted(kwargs))
    
    if keys in save_nametuples:
        my_named_tuple = save_nametuples[keys]
    else:
        my_named_tuple = namedtuple("_".join(keys), keys)
        save_nametuples[keys] = my_named_tuple

    return my_named_tuple(**kwargs)


def named_tupelize(d,ctype):
    keys = sorted(ctype.split("."))
    
    def doit(k):
        if type(k) in [tuple, list]:
            return make_cross(dict(zip(keys, k)))
        elif len(keys) == 1:
            return make_cross({keys[0]:k})
        else: 
            raise Exception("strange case...")
    
    return {
        doit(k):v
        for k,v in d.items()
    }


class Determ:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.define = kwargs
        self.keys = sorted(self.define.keys())
        self.cnt = Cnt(self.dataset, keys=self.keys)

        self._noneKeys = set(k for k,v in self.define.items() if v is None)
        self._notNoneKeys = set(k for k,v in self.define.items() if v is not None)

    def _gen_count(self, typ):
        """
        `typ` is either 'doc' or 'cit'
        Internal function to abstract some code.
        """

        if typ == 'doc':
            c = self.cnt.docs
        elif typ == 'cit':
            c = self.cnt.cits
        else:
            raise Exception('Internal error; see someone who knows something...')

        if len(self._noneKeys):
            def gen():
                for item, value in c.items():
                    bugger = False
                    for k in self._notNoneKeys:
                        if getattr(item, k) != self.define[k]:
                            bugger=True
                            break
                    if bugger:
                        continue

                    ret = {
                        k: getattr(item, k)
                        for k in self._noneKeys
                    }
                    ret['_count'] = value
                    yield ret

            dd = pd.DataFrame(list(gen()))
            dd = dd.sort_values(sorted(self._noneKeys))
            return dd

        else:
            return c[ make_cross(**self.define) ]

    @property
    def docs(self):
        return self._gen_count('doc')
    @property
    def cits(self):
        return self._gen_count('cit')

def ktrans(kwargs):
    # restructure kwargs if need be
    newkwargs = {}
    for k,v in kwargs.items():
        if type(v) == CitedRef:
            v = v.identifier

        if k not in ent.ents:
            try:
                k = getattr(ent, k)
            except AttributeError:
                raise Exception("%s was not a recognized count entity. Use ent.*" % k)

        newkwargs[k] = v
    return newkwargs

# this might just be putting an unnecessary # of steps between the code and its result!
class Dataset:

    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        kwargs = ktrans(kwargs)

        return Determ(
            dataset = self,
            **kwargs
        )


    def items(self, what):
        kwargs = ktrans({what: None})

        process = lambda x:x
        fn_name = "%s_fn" % what
        if hasattr(ent, fn_name):
            process = getattr(ent, fn_name)

        myl = sorted( Determ(
            dataset = self,
            **kwargs
        ).docs[ what ] )
        myl = list(map(process, myl))
        return myl


    def cnt_keys(self):
        avail = Path(env.variable_dir).joinpath(self.name).glob("doc ___ *")
        avail = [x.name for x in avail]
        avail = [x.split("___")[1].strip() for x in avail]
        return sorted(avail)

        
    def variables(self):
        avail = Path(env.variable_dir).joinpath(self.name).glob("*")
        avail_cnts = Path(env.variable_dir).joinpath(self.name).glob("* ___ *")

        avail = set(avail).difference(avail_cnts)
        avail = [x.name for x in avail]
        return sorted(avail)


    def load_variable(self, name):

        # go get this collection, if it isn't already downloaded...
        if not Path(env.variable_dir).joinpath(self.name).exists():
            print("collection", self.name, "does not exist...")
            print("attempting to load from OSF")

            if self.name not in Constants.data_files:
                raise Exception("no data file logged for '%s'"%self.name)

            zip_dest = Path(env.variable_dir, "%s.zip" % self.name)
            if not zip_dest.exists():
                download_file(Constants.data_files[self.name], zip_dest)

            print("Extracting...", str(zip_dest))
            import zipfile
            with zipfile.ZipFile(str(zip_dest), 'r') as zip_ref:
                zip_ref.extractall(str(zip_dest.parent.joinpath(self.name)))

        # otherwise load the pickled file and do it!
        try:
            return pickle.load( Path(env.variable_dir).joinpath(self.name, name).open('rb') )
        except FileNotFoundError:
            raise VariableNotFound((self.name,name))


    def save_variable(self, name, val):
        import pickle
        Path(env.variable_dir).joinpath(self.name).mkdir(exist_ok=True)
        pickle.dump( val, Path(env.variable_dir).joinpath(name).open('wb') )




class Cnt:
    cache = {}

    def __init__(self, dataset, keys=[]):
        self.dataset = dataset

        if type(keys) != list:
            keys = [keys]

        k = ".".join(sorted(keys))

        if (dataset.name,k) not in self.cache:
            varname_d = "doc ___ %s" % k
            varname_c = "ind ___ %s" % k

            #print(k)
            self.docs = defaultdict(int, named_tupelize( dict(dataset.load_variable(varname_d)), k ))
            self.cits = defaultdict(int, named_tupelize( dict(dataset.load_variable(varname_c)), k ))
            self.cache[(dataset.name,k)] = (self.docs, self.cits)
        else:
            self.docs, self.cits = self.cache[(dataset.name,k)]





    def collapse(self, filt=lambda x:True, on=None):
        new_c = defaultdict(int)
        for item, count in c.items():
            if not f(item):
                continue

            new_c[on(item)] += count

        return Counter( dict(new_c.items()) )