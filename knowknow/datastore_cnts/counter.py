from .. import defaultdict, Path, np, env
from .count_cache import Dataset
from ..datasources.wos import doc_iterator
import pickle


#from scipy.sparse import lil_matrix
#from scipy.sparse import DOK


from csv import DictReader
# This cell ensures there are not overflow errors while importing large CSVs

import sys
import csv
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)







class wos:
    
    def __init__(
        self, 
        wos_txt_base, output_database, name_blacklist=[], 
            RUN_EVERYTHING=True, 
            groups=None, group_reps=None,
            citations_filter=None, journals_filter=None, debug=False, 
            trimCounters = False, wos_type='csv'
    ):
        
        self.wos_txt_base = Path(wos_txt_base)
        assert(self.wos_txt_base.exists())
        
        self.name_blacklist = name_blacklist
        self.debug = debug
        self.output_database = output_database
        self.groups = groups
        self.group_reps = group_reps

        self.wos_type = wos_type
        
        self.RUN_EVERYTHING = RUN_EVERYTHING
        self.citations_filter = citations_filter
        self.journals_filter = journals_filter
        self.trimCounters = trimCounters
        
        
        
        # Instantiating counters
        self.ind = defaultdict(lambda:defaultdict(int))
        self.track_doc = defaultdict(lambda:defaultdict(set))
        self.doc = defaultdict(lambda:defaultdict(int))


    
    def extra_filter_doc_ref(self, doc,ref):
        return True
        
        
        
    
    def count_cocitations(self):
        
        allowed_refs = Counter(dict(self.doc['c'].items())).most_common(1000)
        allowed_refs = set( x[0] for x in allowed_refs )

        print("# allowed references for cocitation analysis: %s" % len(allowed_refs))
        print("Examples: %s" % str(list(allowed_refs)[:3]))
        
        refcount = 0
        
        for doc in self.doc_iterator():
            refs = [r for r in doc.generate_references() if self.extra_filter_doc_ref(doc,r)]
            for ref1 in refs:
                if ref1.full_ref not in allowed_refs:
                    continue
                    
                for ref2 in refs:
                    if ref2.full_ref not in allowed_refs:
                        continue
                    if ref1.full_ref >= ref2.full_ref:
                        continue

                    self.cnt((ref1.full_ref,ref2.full_ref), 'c1.c2', doc.uid)
                    self.cnt((ref1.publish,ref2.publish), 'c1y.c2y', doc.uid)

                    refcount += 1
                    if refcount % 10000 == 0:
                        print("%s cocitations logged" % refcount)

        print("Finished cocitation logging!")
        
        


            
        
    def cnt(self, term, space, doc):
        if ".".join(sorted(space.split("."))) != space:
            raise Exception(space, "should be sorted...")

        if type(term) != tuple:
            term = tuple([term])

        # it's a set, yo
        self.track_doc[space][term].add(doc)
        # update cnt_doc
        self.doc[space][term] = len(self.track_doc[space][term])
        # update ind count
        self.ind[space][term] += 1


        

    def doc_iterator(self):
        it = doc_iterator(base=self.wos_txt_base, type=self.wos_type)

        for doc in it:
            # implements the journals filter!
            if self.journals_filter is not None and doc.journal.lower() not in self.journals_filter:
                continue

            yield doc



    def count_citation(self, doc, ref):
        
        full_ref = ref.full_ref
        
        # implement grouping of references
        if self.groups is not None:
            if full_ref in self.groups:
                # retrieves retrospectively-computed groups
                full_ref = self.group_reps[
                    self.groups[full_ref]
                ]
            else:
                raise Exception('not in group...', full_ref)

        ref.full_ref = full_ref
        
        self.cnt(doc.journal, 'fj', doc.uid)
        self.cnt(doc.publish, 'fy', doc.uid)
        self.cnt(ref.publish, 'ty', doc.uid)
        self.cnt((ref.full_ref, doc.publish), 'c.fy', doc.uid)
        self.cnt((ref.full_ref, doc.journal), 'c.fj', doc.uid)

        self.cnt((doc.journal, doc.publish), 'fj.fy', doc.uid)

        self.cnt(ref.full_ref, 'c', doc.uid)
        
        if self.RUN_EVERYTHING:
            
            self.cnt((doc.publish, ref.publish), 'fy.ty', doc.uid)
            self.cnt((doc.journal, ref.publish), 'fj.ty', doc.uid)
                    
            self.cnt(ref.author, 'ta', doc.uid)
            self.cnt((doc.publish,ref.author), 'fy.ta', doc.uid)
            self.cnt((doc.journal,ref.author), 'fj.ta', doc.uid)
            
            self.cnt((ref.full_ref, doc.journal, doc.publish), 'c.fj.fy', doc.uid)
            
            # first author!
            ffa = doc.citing_authors[0]
            self.cnt(ffa, 'ffa', doc.uid)
            self.cnt((ffa,doc.publish), 'ffa.fy', doc.uid)
            self.cnt((ffa,doc.journal), 'ffa.fj', doc.uid)
            self.cnt((ref.full_ref,ffa), 'c.ffa', doc.uid)
            #self.cnt((ref.full_ref,ffa, doc.publish), 'c.ffa.fy', doc.uid)
            #self.cnt((ffa,r['SO'], int(r['PY'])), 'ffa.fj.fy', doc.uid)

            for a in doc.citing_authors:
                self.cnt(a, 'fa', doc.uid)
                self.cnt((a, doc.publish), 'fa.fy', doc.uid)
                self.cnt((a, doc.journal), 'fa.fj', doc.uid)
                #self.cnt((ref.full_ref,a, doc.publish), 'c.fa.fy', doc.uid)
                #self.cnt((a,r['SO'], int(r['PY'])), 'fa.fj.fy', doc.uid)

                self.cnt((ref.full_ref,a), 'c.fa', doc.uid)

    def main_loop(self):
        
        for doc in self.doc_iterator():
            
            for ref in doc.generate_references():
                
                if ref.author in self.name_blacklist:
                    continue
                    
                if self.groups is not None and ref.full_ref not in self.groups:
                    continue

                if not self.extra_filter_doc_ref(doc,ref):
                    continue
                
                self.count_citation(
                    doc,
                    ref
                )
    
    def save_counters(self):
        db = Dataset(self.output_database)
        db.clear_all()
        for k,count in self.doc.items():
            varname = "doc ___ %s" % k
            db.save_variable(varname, dict(count))

            
        for k,count in self.ind.items():
            varname = "ind ___ %s" % k
            db.save_variable(varname, dict(count))
    

    def count_ages(self):
        
        cbirthdays = defaultdict(lambda:2050)
        for (c,y),count in self.doc['c.fy'].items():
            if count == 0:
                continue
            cbirthdays[c] = min(cbirthdays[c], y)
            
        ffabirthdays = defaultdict(lambda:2050)
        for (c,y),count in self.doc['ffa.fy'].items():
            if count == 0:
                continue
            ffabirthdays[c] = min(ffabirthdays[c], y)

        tabirthdays = defaultdict(lambda:2050)
        for (y,c),count in self.doc['fy.ta'].items():
            if count == 0:
                continue
            tabirthdays[c] = min(tabirthdays[c], y)

        for doc in self.doc_iterator():

            refs = list(doc.generate_references())
            for i, ref in enumerate(refs):

                if ref.author in self.name_blacklist:
                    continue

                if self.groups is not None and ref.full_ref not in self.groups:
                    continue

                if not self.extra_filter_doc_ref(doc,ref):
                    continue

                c = ref.full_ref
                ffa = doc.citing_authors[0]
                ta = ref.author

                # skips the hitherto uncounted
                if c not in self.doc['c'] or self.doc['c'][c] == 0:
                    continue
                if ffa not in self.doc['ffa'] or self.doc['ffa'][ffa] == 0:
                    continue
                if ta not in self.doc['ta'] or self.doc['ta'][ta] == 0:
                    continue

                cage1 = doc.publish - cbirthdays[c]
                ffaage1 = doc.publish - ffabirthdays[ffa]
                taage1 = doc.publish - tabirthdays[ta]

                if not all(x>=0 for x in [cage1,ffaage1,taage1]):
                    print(cage1, ffaage1, taage1)
                    print(self.doc['c'][ref.full_ref], self.doc['ffa'][doc.citing_authors[0]], self.doc['ta'][ref.author])
                    print(ref.full_ref, doc.citing_authors[0], ref.author)
                    print(cage1, ffaage1, taage1)
                    raise Exception('this is impossible... age stuff')

                self.cnt((cage1,doc.journal), 'cAge.fj', doc.uid)

                self.cnt((cage1,doc.citing_authors[0]), 'cAge.ffa', doc.uid)
                for author in doc.citing_authors:
                    faage1 = doc.publish - ffabirthdays[author]
                    self.cnt((cage1,author), 'cAge.fa', doc.uid)
                    self.cnt(faage1, 'faAge', doc.uid)

                    self.cnt((faage1,ref.author), 'faAge.ta', doc.uid)
                    self.cnt((faage1,doc.publish,ref.author), 'faAge.fy.ta', doc.uid)
                    self.cnt((faage1,taage1), 'faAge.taAge', doc.uid)
                    self.cnt((cage1,faage1), 'cAge.faAge', doc.uid)

                    self.cnt((faage1,doc.publish), 'faAge.fy', doc.uid)

                self.cnt(ffaage1, 'ffaAge', doc.uid)

                self.cnt((ffaage1,ref.author), 'ffaAge.ta', doc.uid)
                self.cnt((ffaage1,doc.publish,ref.author), 'ffaAge.fy.ta', doc.uid)
                self.cnt((ffaage1,taage1), 'ffaAge.taAge', doc.uid)
                self.cnt((cage1,ffaage1), 'cAge.ffaAge', doc.uid)

                self.cnt((cage1,doc.publish), 'cAge.fy', doc.uid)
                self.cnt((doc.publish,taage1), 'fy.taAge', doc.uid)
                self.cnt((ffaage1,doc.publish), 'ffaAge.fy', doc.uid)

                for j, ref2 in enumerate(refs):
                    cage2 = doc.publish - cbirthdays[ref2.full_ref]
                    if i >= j:
                        continue

                    self.cnt((cage1, cage2), 'c1age.c2age', doc.uid)


    def count_coauthors(self):
        
        for doc in self.doc_iterator():
            for a1 in doc.citing_authors:
                for a2 in doc.citing_authors:
                    if a1 != a2:
                        self.cnt((a1, a2), 'fa1.fa2', doc.uid)

        print("Finished coauthor logging!")


    def run(self):
        self.main_loop()
        if self.RUN_EVERYTHING:
            self.count_ages()
            self.count_cocitations()
            self.count_coauthors()
        self.save_counters()

def matches(name, dir=None):
    if dir is None:
        dir='.'
    dir = Path(dir)

    fns = dir.glob(f"{name}*.pickle")
    fns = [x.name for x in fns]
    fns = [x.replace(".counts.pickle","") for x in fns]
    fns = [x.replace(".ids.pickle","") for x in fns]
    fns = set(fns)
    return fns
    
if False:


    class counters:

        def __init__(self, name):
                    
            fns = Path('.').glob(f"{name}*.pickle")
            fns = [x.name for x in fns]
            fns = [x.replace(".counts.pickle","") for x in fns]
            fns = [x.replace(".ids.pickle","") for x in fns]
            fns = set(fns)

            print("Loading", ", ".join(fns))

            self.counters = {
                fn: counter(fn)
                for fn in fns
            }

        def __getattribute__(self, what):
            return print(what)


class counter:
    """
    The counter class manages coocurrence counts.
    It is useful for storing and retrieving them with memory and disk-space efficiently,
        and for containing maintenance and analysis functions.
    It is written at relatively high level of generality, 
        to reduce overall code size.
    Initiate a counter with its name, 
        which identifies it in the Dataset.
    """
    def __init__(self, name=None, typ="ddict", paradigm='one_file', home_dir=None):
        import pickle

        self.name = name
        self.typ = typ
        self.paradigm = paradigm

        self._sortsave = {}
        
        self.counts = {}
        self.ids = {}
        self.names = {}

        self.home_dir = home_dir
        if self.home_dir is None:
            self.home_dir = env.variable_dir
        self.home_dir = Path(self.home_dir)
        
        self.path = self.home_dir.joinpath(self.name)
        self.path.mkdir(exist_ok=True)

        self.meta = {}
        ap = self.path / '_attributes'
        if ap.exists():
            with ap.open('rb') as inf:
                self.meta = pickle.load(inf)
        else:
            with ap.open('wb') as outf:
                pickle.dump(self.meta, outf)

        if self.name is not None:
            fn = self.name

            if (self.home_dir / self.name).is_dir():
                ctf = self.home_dir / self.name / f'counts.pickle'
                idf = self.home_dir / self.name / f'ids.pickle'
            else:
                ctf = self.home_dir / f'{fn}.counts.pickle'
                idf = self.home_dir / f'{fn}.ids.pickle'
        
            ex1,ex2 = False,False
            try:
                ex1,ex2 = Path(ctf).exists(), Path(idf).exists()
            except OSError: # wtf windows??
                pass
                    
            if self.typ == 'idmap':
                if not ex2:
                    print("no id.pickle file // setting typ='ddict'")
                    typ = 'ddict'

                with open(idf, 'rb') as inf:
                    self.ids = pickle.load(inf)

                self.names = {
                    cname: {i:n for n,i in myids.items()}
                    for cname,myids in self.ids.items()
                }

            if self.paradigm == 'one_file':
                
                if Path(ctf).exists():
                    print(f'Loading {name} from disk...')
                    with open(ctf, 'rb') as inf:
                        self.counts = pickle.load(inf)

                else:
                    ms = matches(self.name)
                    if len(ms):
                        print('Did you mean ', ", ".join(ms))
                    raise Exception('file not found', ctf)

            elif self.paradigm == 'many_file':
                # don't load anything, it's lazy loaded now...
                pass

            else:
                raise Exception("use an acceptable paradigm. this will bust code")


    def no_duplicate(self): # this doesn't copy the data structures...
        c = counter()
        c.counts = self.counts
        c.ids = self.ids
        c.names = self.names
        return c

    def loadinit(self, c):
        c = tuple(sorted(c))

        if c not in self.counts:

            # lazy loading for multi-file setup
            if self.paradigm == 'many_file' and self.name is not None:
                typ = '.'.join(c)
                print(f'loading count file {self.name} / {typ}')
                if (self.home_dir / self.name).is_dir():
                    ctf = self.home_dir / self.name / f'counts~{typ}.pickle'
                else:
                    ctf = self.home_dir / f'{self.name}.counts~{typ}.pickle'

                if Path(ctf).exists():
                    with open(ctf, 'rb') as inf:
                        self.counts[c] =  pickle.load(inf)                
                    return
                else:
                    raise Exception('file not found', ctf)

            #self.counts[c] = np.zeros( tuple([100]*len(c)), dtype=object ) # initialize small N-attrs sized array... agh np.int16 overflows easily... gotta upgrade to object?
            #self.counts[c] = DOK( tuple([100]*len(c)) )
            # init some zeros if it doesn't exist
            self.counts[c] = defaultdict(int)
        
                
    ###############   FOR COUNTING   ####################
    
    def count(self, info, combinations=[], scale=1):
        """
        `info` is a dictionary of information about the entity.
        `combinations` is a list of the combinations of keys in `info` which the counter should count
        """

        if False: # put this in an except phrase somewhere if you really want it here
            ask = set(y for x in combinations for y in x)
            have = set(info)

            if not ask.issubset( have ):
                ask_s = ", ".join( sorted( ask ) )
                have_s = ", ".join( sorted( have ) )
                raise Exception(f'Asked to count {ask_s}, have {have_s}.')

        if self.typ == "idmap":

            # do once for many combos
            ids = {
                k:self.idof( k, v )
                for k,v in info.items()
            }
            
            for c in combinations:
                if c not in self._sortsave:
                    self._sortsave[c] = tuple(sorted(c)) # consistency

                c = self._sortsave[c]
                # dim = len(c)
                # cname = ".".join(sorted(c))
                
                self.loadinit(c)
                self.counts[c][ tuple( ids[ck] for ck in c ) ] += scale

        elif self.typ == "ddict":

            for c in combinations:
                # dim = len(c)
                # cname = ".".join(sorted(c))
                
                if c not in self._sortsave:
                    self._sortsave[c] = tuple(sorted(c)) # consistency

                c = self._sortsave[c]
                # dim = len(c)
                # cname = ".".join(sorted(c))
                
                self.loadinit(c)
                self.counts[c][ tuple( info[ck] for ck in c ) ] += scale

    # returns id, or makes a new one.
    # often needs to expand the arrays
    def idof(self, kind, name):
            
        # if you've never seen it, add this ID to the dict
        if name not in self.ids[kind]:
            new_id = len(self.ids[kind])
            self.ids[kind][name] = new_id
            self.names[kind][new_id] = name
            
            if False:
                # now have to expand all the np arrays...
                # but only if they're arrays!
                # have to loop through all series, because you have to expand fy.ty.t as well as t :O
                for k in self.counts:
                    if not kind in k: # remember k is a tuple... kind is a string representing a type of count
                        continue
                        
                    arr_index = k.index(kind)
                    
                    """
                    # no need to pad if it's already big enough
                    current_shape = self.counts[k].shape[arr_index]
                    if current_shape >= new_id+1:
                        continue
                    """
                        
                    """
                    self.counts[k] = np.pad( 
                        self.counts[k], 

                        # exponential growth :) 
                        # this (current_shape*0.25) limits the number of pad calls we need.
                        # these become expensive if we do them all the time
                        [(0,int(current_shape*0.25)*(dim==arr_index)) for dim in range(len(k))], 

                        mode='constant', 
                        constant_values=0 
                    )
                    """

                    """
                    self.counts[k].resize((
                        current_shape*( 1 + 0.25*(dim==arr_index))
                        for dim in range(len(k))
                    ))
                    """
                    
                    """
                    self.counts[k].shape = tuple(
                        int( current_shape*( 1 + 0.25*(dim==arr_index)) )
                        for dim in range(len(k))
                    )
                    """
            
        return self.ids[kind][name]

    def nameof(self, kind, id):
        return self.names[kind][id]
                
                
    ###############   FOR RECALL   ####################
        
    def __call__(self, *args, **kwargs):
        cname = tuple(sorted(kwargs.keys()))
        self.loadinit(cname)
        noneKeys = {k for k,v in kwargs.items() if v is None}
        notNoneKeys = {k for k,v in kwargs.items() if v is not None}

        if len(noneKeys):
            def gen():
                        
                #wh = np.where( self.counts[cname] > 0 )
                for parts in self.counts[cname]:
                    myc = self.counts[cname][parts]
                    nameparts = {kind:self.names[kind][p] for kind,p in zip(cname, parts)}

                    bugger = False
                    # add up all the parts that make sense...
                    for k in notNoneKeys:
                        if nameparts[k] != kwargs[k]:
                            bugger=True
                            break
                    if bugger:
                        continue

                    ret = {
                        k: nameparts[k]
                        for k in noneKeys
                    }

                    ret['count'] = myc
                    yield ret

            import pandas as pd
            final = pd.DataFrame(list(gen()))
            final = final.sort_values(sorted(noneKeys))
            return final
        
        if self.typ == "idmap":
            # just figuring out what the proper index in the matrix is...
            my_id = []
            for cpart in cname:
                if kwargs[cpart] not in self.ids[ cpart ]:
                    return 0
                my_id.append( self.ids[ cpart ][ kwargs[cpart] ] )

            my_id = tuple(my_id)

            return self.counts[ cname ][ my_id ]

        elif self.typ == "ddict":

            return self.counts[cname][tuple(
                kwargs[cn] for cn in cname
            )]

        
        

    def keys(self, typ):
        return list(self.names[typ])
    
    def items(self, *typs):
        from .. import make_cross

        order = sorted(range(len(typs)), key=lambda x:typs[x])
        typsk = tuple(typs[i] for i in order)
        self.loadinit(typsk)


        for item, c in self.counts[ typsk ].items():
            item = [item[o] for o in order]
            
            if False:
                t = make_cross({
                    t: self.names[t][ item[ typsk.index(t) ] ] 
                    for ti,t in enumerate(typs)
                })
            else:
                if self.typ == 'idmap':
                    t = tuple(
                        self.names[typ][name] 
                        for typ,name in zip(typsk, item)
                    )
                elif self.typ == 'ddict':
                    t = tuple(item)

                else:
                    raise Exception("invalid typ")

            if len(t) == 1:
                t = t[0]
            yield (t, c)

        return

        # this also should work fine, but is a bit complex
        from itertools import product
        typs = sorted(typs)
        parts = {
            typ: sorted(self.ids[typ].items())
            for typ in typs
        }

        for item in product(*[parts[t] for t in typs]):
            print(item)
            if self( **dict(zip(typs, [x[0] for x in item])) ) > 0:
                yield [x[0] for x in item]


    def trend(self, dtype, name, years=None):
        from .time_trend import TimeTrend
        return TimeTrend(
            dtype = dtype,
            name = name,
            dataset = self,
            years = years
        )

    def drop(self, typ):
        """
        Deletes all counts of a specific type.
        Useful to conserve memory
        """

        del self.counts[typ]
        del self.names[typ]
        del self.ids[typ]

    def dropall(self, typ):
        """
        Drops anything having to do with a certain type...
        """

        todrop = []
        for k in self.counts:
            if any(kp == typ for kp in k):
                todrop.append(k)
        
        for td in todrop:
            self.drop(td)


    
    def _old_delete(self, typ, which):        
        for cnames in self.counts:
            if typ not in cnames:
                continue
                
        del_idx = set(self.ids[typ][i] for i in which)
        del_idx_np = np.array(sorted(del_idx))
        keep_idx = set(self.ids[typ].values()).difference(del_idx)
        keep_idx_np = np.array(sorted(keep_idx))
        
        print(f"Deleting {len(del_idx)/1e6:0.1f}M. Leaving {len(keep_idx):,}.")

        new_ids = {}
        new_id_i = 0
        for t,i in sorted( self.ids[typ].items(), key=lambda x:x[1] ):
            if i not in keep_idx:
                continue

            new_ids[t] = new_id_i
            new_id_i += 1
            
        self.ids[typ] = new_ids
        
        for ckey, c in self.counts.items():
            if typ not in ckey:
                continue

            cur_count = self.counts[ckey]
            del_index = ckey.index( typ )
            self.counts[ckey] = np.delete( cur_count, del_idx_np, axis=del_index )
        
    def _old_prune_zeros(self):
        typs = self.ids.keys()
        
        for typ in typs:
            base_count = self.counts[(typ,)]
            zero_idx = [ ti for ti,c in enumerate(base_count) if c == 0 ]
            if not len(zero_idx):
                continue

            del_start = min(zero_idx)

            del_cols = np.arange(del_start, base_count.shape[0])

            for ckey, c in self.counts.items():
                if typ not in ckey:
                    continue

                cur_count = self.counts[ckey]
                del_index = ckey.index( typ )
                self.counts[ckey] = np.delete( cur_count, del_cols, axis=del_index )
                
    def save(self, name=None, typ=None):

        if name is None:
            if self.name is None:
                raise Exception('please give a name for the dataset')
            else:
                name = self.name

        if self.paradigm == 'one_file':
            with open(self.home_dir / f'{name}.counts.pickle', 'wb') as outf:
                pickle.dump(self.counts, outf)

            with open(self.home_dir / f'{name}.ids.pickle', 'wb') as outf:
                pickle.dump(self.ids, outf)

        elif self.paradigm == 'many_file':
            for typ, cc in self.counts.items():
                typ = '.'.join(sorted(typ))
                with open(self.home_dir / f'{name}.counts~{typ}.pickle', 'wb') as outf:
                    pickle.dump(cc, outf)

            if self.typ == 'idmap':
                with open(self.home_dir / f'{name}.ids.pickle', 'wb') as outf:
                    pickle.dump(self.ids, outf)
            
    def summarize(self):
        print( [(k, c.shape) for k,c in self.counts.items()])


    # ///////////    variable functions   //////////

    def save_variable(self, name, val):
        import pickle
        pickle.dump( val, self.path.joinpath(name).open('wb') )

    def load_variable(self, name):
        with open(self.path.joinpath( name ), 'rb') as inf:
            return pickle.loads( inf )


    # ///////////    counter metadata    ///////////
    
    def __setitem__(self, k, v):
        self.meta[k] = v
        self.save_variable('_attributes', self.meta)

    def __getitem__(self, k):
        return self.meta[k]

    def __contains__(self, k):
        return k in self.meta