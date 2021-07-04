from .. import env, defaultdict, pd

import pickle
from .. import Path, Counter, defaultdict, download_file, VariableNotFound

__all__ = [
    'Dataset','CitedRef'
]

"""
import sys
#sys.path.append('C:/Users/amcga/GitHub/knowknow/archive')
sys.path.append('C:/Users/amcga/GitHub/knowknow')
sys.path = sys.path[1:]
print(sys.path)
from knowknow.datastore_cnts.count_cache import *
wos = Dataset('sociology-wos')
wos(fy=1993)
"""

class CitedRef:
    def __init__(self, identifier):
        self.identifier = identifier

        isplit = self.identifier.split("|")

        self.full_ref = self.identifier
        self.author = isplit[0]
        self.pyear = None
        if len(isplit) == 2:
            self.type = 'book'
            self.title = isplit[1]
        else:
            self.type = 'article'
            self.pyear = int(isplit[1])
            self.title = isplit[2]

    def __repr__(self):
        return "< %s >" % "Cited %s: %s" % (self.type, self.full_ref)

class ent:
    citing_author = 'fa'
    cited_author = 'ta'
    publication_year = 'fy'
    cited_year = 'ty'
    cited_pub = 'c'
    citing_journal = 'fj'
    first_citing_author = 'ffa'

    #@classmethod
    #def c_fn(cls, x):
    #    return CitedRef(x)

    ents = 'fa,ta,fy,ty,c,fj,ffa'.split(",")



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
    data_files = {}


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
        self.cnt = Cnt.by(self.dataset, *self.keys)

        self._noneKeys = set(k for k,v in self.define.items() if v is None)
        self._notNoneKeys = set(k for k,v in self.define.items() if v is not None)

    def refine(self, **kwargs):
        new_args = dict(self.define, **kwargs)
        return Determ(self.dataset, **new_args)


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

                    # this block just skips the non-representative version of C
                    if False:
                        if self.dataset.groups is not None and 'c' in self.define: #DELETED
                            if item.c in self.dataset.groups:
                                myg = self.dataset.groups[ item.c ]
                                #print('grouping!', myg, item.c)
                                if item.c != self.dataset.group_reps[ myg ]:
                                    continue

                    bugger = False
                    for k in self._notNoneKeys:
                        #if type(self.define[k]) in {list,set,range}:
                        #    if getattr(item, k) not in self.define[k]:
                        #        bugger=True
                        #        break
                        #else:
                        if getattr(item, k) != self.define[k]:
                            bugger=True
                            break
                    if bugger:
                        continue

                    ret = {
                        k: getattr(item, k)
                        for k in self._noneKeys
                    }

                    ret['count'] = value

                    if False:
                        # this part actually sums over the values for this item.
                        if self.dataset.groups is not None and 'c' in self.define and item.c in self.dataset.groups: #DELETED
                            myg = self.dataset.groups[item.c]

                            to_set = [k for k,g in self.dataset.groups.items() if g == myg]
                            #print(to_set)
                                
                            count = 0
                            for ts in to_set:
                                #print(item)
                                item_search = item.__class__( *item )
                                item_search = item_search._replace(c=ts)
                                #print(item_search)
                                count += c[ item_search ]
                                #print(item_search, count)

                            ret['count'] = count

                    else:
                        ret['count'] = value

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

    attr_cache = {}

    def __init__(self, name):
        self.name = name

        #try:
        #    self.groups = self.load_variable('groups')
        #except VariableNotFound:
        #    self.groups = None
        #try:
        #    self.group_reps = self.load_variable('group_reps')
        #except VariableNotFound:
        #    self.group_reps = None



        self.groupings = []
        

        try:
            groups = self.load_variable('groups')
            group_reps = self.load_variable('group_reps')
            self.groupings.append(
                ('c', {k: group_reps[v] for k,v in groups.items()})
            )
        except VariableNotFound:
            pass
        

        self.load_attributes()





    def __call__(self, typ=None, kwarg_dict=None, **kwargs):

        if kwarg_dict is not None:
            kwargs = kwarg_dict
            
        det = Determ(
            dataset = self,
            **kwargs
        )
        if typ is None:
            return det
        else:
            return getattr(det,typ)

        return
        assert(typ in {'cit','doc'})

        keys = list(kwargs.keys())
        count = self.by_raw(keys)

        if typ == 'cits':
            count = count.cits
        if typ == 'docs':
            count = count.docs

        none_keys = [k for k,v in kwargs.items() if v is None]
        kwargs = ktrans(kwargs)

        return Determ(
            dataset = self,
            **kwargs
        )




    def search(self, dtype, name):
        if dtype not in {'fa','ta','c'}:
            raise Exception('datatype error')

        assert(type(name) == str)

        return [x for x in self.items(dtype) if name.lower() in x.lower() and x.lower().index(name.lower())==0]






    def save_figure(self, name):
        from .. import plt
        outdir = Path(env.figure_dir)
        if not outdir.exists():
            outdir.mkdir()
        print("Saving to '%s'"%outdir)
        plt.savefig(str(outdir.joinpath("%s.png" % name)), bbox_inches="tight")



    def display_figure(self, x, titles=True):
        from .. import plt, display, HTML, Image
        displayed_normally = set()

        import os
        import time
        from datetime import datetime
            
        search = list(Path(env.figure_dir).glob(x))
        if not len(search):
            raise Exception("What figure!? '%s'" % x)
        
        for f in sorted(search):
            if titles:
                display(HTML("<h1>%s</h1>" % (f.name)))
            display(Image(filename=f))
            displayed_normally.add(f.name)
                
        return displayed_normally







    def exists(self):
        return Path(env.variable_dir).joinpath(self.name).exists()

    def items(self, what):
        kwargs = ktrans({what: None})

        process = lambda x:x
        fn_name = "%s_fn" % what
        if hasattr(ent, fn_name):
            process = getattr(ent, fn_name)

        myl = sorted( Determ(
            dataset = self,
            **kwargs
        ).cits[ what ] )
        myl = list(map(process, myl))
        return myl

    @property
    def start(self):
        return self['RELIABLE_DATA_STARTS_HERE']
        
    @property
    def end(self):
        return self['RELIABLE_DATA_ENDS_HERE']

    def cnt_keys(self):
        avail = Path(env.variable_dir).joinpath(self.name).glob("doc ___ *")
        avail = [x.name for x in avail]
        avail = [x.split("___")[1].strip() for x in avail]
        return sorted(avail)

    def trend(self, dtype, name):
        from .time_trend import TimeTrend
        return TimeTrend(
            dtype = dtype,
            name = name,
            dataset=self
        )

    def upload(self, api_key, 
        dataset_description="", 
        dataverse_name = 'kk-citation-counts',
        dataverse_server = 'https://dataverse.harvard.edu'):

        from ..dataverse import upload

        return upload( 
            api_key, 
            self.name,
            dataset_description=dataset_description, 
            dataverse_name=dataverse_name,
            dataverse_server=dataverse_server)
        
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
                raise VariableNotFound(("no data file", self.name))

            zip_dest = Path(env.variable_dir, "%s.zip" % self.name)
            if not zip_dest.exists():
                download_file(Constants.data_files[self.name], zip_dest)

            print("Extracting...", str(zip_dest))
            import zipfile
            with zipfile.ZipFile(str(zip_dest), 'r') as zip_ref:
                zip_ref.extractall(str(zip_dest.parent.joinpath(self.name)))

        # otherwise load the pickled file and do it!
        try:
            print('loading variable %s/%s from disk' % (self.name,name))
            return pickle.load( Path(env.variable_dir).joinpath(self.name, name).open('rb') )
        except FileNotFoundError:
            raise VariableNotFound((self.name,name))

    def clear_all(self):
        # remove the files
        for f in Path(env.variable_dir).joinpath(self.name).glob("*"):
            f.unlink()

    def save_variable(self, name, val):
        import pickle
        outdir = Path(env.variable_dir).joinpath(self.name)
        outdir.mkdir(exist_ok=True)
        pickle.dump( val, outdir.joinpath(name).open('wb') )

    def load_attributes(self):
        if self.name in self.attr_cache:
            self.docs = self.attr_cache[self.name]

        else:
            try:
                self.docs = self.load_variable('_attributes')
            except VariableNotFound:
                self.docs = {}

            self.attr_cache[self.name] = self.docs

    def __getitem__(self, what):
        return self.docs[what]
        
    def set_attribute(self, what, value):
        self.docs[what] = value
        self.save_variable('_attributes', self.docs)

    def add_grouping(self, typ, mp):
        self.groupings.append( (typ,mp) )

    def by_raw(self, *args):
        return Cnt.by(self, *args) # passes this dataset as the first argument
        
    def by(self, *args):
        return Cnt.by(self, *args) # passes this dataset as the first argument






class Cnt:
    cache = {}

    def __init__(self, dataset, keys=[]):

        if type(keys) != list:
            keys = [keys]
        self.keys = keys
        
        self.dataset = dataset
        
        self._docs = None
        self._cits = None


    @classmethod
    def by(cls, dataset, *keys):
        if not len(keys):
            raise Exception('by what?')
        if type(keys[0]) == list:
            return cls.by(*keys[0])
            
        keys = list(keys)

        myk = (dataset.name, frozenset(keys))

        if myk not in cls.cache:
            ret = Cnt(dataset, keys)
            cls.cache[myk] = ret

        return cls.cache[myk]
        
    def load_counts(self, varname, k):
    
        ret = defaultdict(int)
        # populate _cits, keeping grouping in mind
        basic_counts = named_tupelize( dict(self.dataset.load_variable(varname)), k )
        for k,v in basic_counts.items():
            
            # replacements by grouping
            #salty_element = False

            for typ, mp in self.dataset.groupings:
                #print(k)
                if hasattr(k, typ):
                    #print(typ, len(mp))
                    
                    current_attr = getattr(k, typ)
                    #print(current_attr)
                    if current_attr not in mp:
                        #salty_element = True
                        continue

                    k = k._replace(**{typ: mp[current_attr]})
                    #print(k)
            
            #if salty_element:
            #    continue

            ret[ k ] += v
        
        return ret

    @property
    def cits(self):
        if self._cits is None:
            k = ".".join(sorted(self.keys))
            varname = "ind ___ %s" % k
            self._cits = self.load_counts(varname, k)
        
        return self._cits

    @property
    def docs(self):
        if self._docs is None:
            k = ".".join(sorted(self.keys))
            varname = "doc ___ %s" % k
            self._docs = self.load_counts(varname, k)
        
        return self._docs


    def collapse(self, filt=lambda x:True, on=None):
        new_c = defaultdict(int)
        for item, count in self._docs.items():
            if not filt(item):
                continue

            new_c[on(item)] += count

        return Counter( dict(new_c.items()) )