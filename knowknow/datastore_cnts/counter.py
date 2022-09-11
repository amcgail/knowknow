from .. import defaultdict, Path
from .count_cache import Dataset
from ..datasources.wos import doc_iterator






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
    def __init__(self, name=None):
        import pickle
        self.name = name
        
        ctf = f'{name}.counts.pickle'
        idf = f'{name}.ids.pickle'
        
        ex = False
        try:
            ex = Path(ctf).exists()
        except OSError: # wtf windows??
            pass

        if ex:
            print(f'Loading {name} from disk...')
            with open(ctf, 'rb') as inf:
                self.counts = pickle.load(inf)
            with open(idf, 'rb') as inf:
                self.ids = pickle.load(inf)
                
        else:
            if name is not None:
                print(f'Blank counter with name {name}')
            else:
                print(f'Blank counter with no name')
                
            self.counts = {}
            self.ids = {}
                
    ###############   FOR COUNTING   ####################
    
    def count(self, info, combinations=[]):
        """
        `info` is a dictionary of information about the entity.
        `combinations` is a list of the combinations of keys in `info` which the counter should count
        """
        ids = {
            k:self.idof( k, v )
            for k,v in info.items()
        }
        
        for c in combinations:
            # dim = len(c)
            # cname = ".".join(sorted(c))
            assert(all( x in info for x in c ))
            
            if c not in self.counts:
                self.counts[c] = np.zeros( tuple([10]*len(c)), dtype=object ) # initialize small N-attrs sized array... agh np.int16 overflows easily... gotta upgrade to object?

            self.counts[c][tuple( ids[ck] for ck in c )] += 1

    # returns id, or makes a new one.
    # often needs to expand the arrays
    def idof(self, kind, name):       
        
        if kind not in self.ids:
            self.ids[kind] = {}
            
        # if you've never seen it, add this ID to the dict
        if name not in self.ids[kind]:
            new_id = len(self.ids[kind])
            self.ids[kind][name] = new_id
            
            # now have to expand all the np arrays...
            # have to loop through all series, because you have to expand fy.ty.t as well as t :O
            for k in self.counts:
                if not kind in k: # remember k is a tuple... kind is a string representing a type of count
                    continue
                    
                arr_index = k.index(kind)
                
                # no need to pad if it's already big enough
                current_shape = self.counts[k].shape[arr_index]
                if current_shape >= new_id+1:
                    continue
                    
                self.counts[k] = np.pad( 
                    self.counts[k], 

                    # exponential growth :) 
                    # this (current_shape*0.25) limits the number of pad calls we need.
                    # these become expensive if we do them all the time
                    [(0,int(current_shape*0.25)*(dim==arr_index)) for dim in range(len(k))], 

                    mode='constant', 
                    constant_values=0 
                )
            
        return self.ids[kind][name]
        
          
                
                
    ###############   FOR RECALL   ####################
        
    def __call__(self, *args, **kwargs):
        cname = tuple(sorted(kwargs.keys()))
        
        # just figuring out what the proper index in the matrix is...
        my_id = []
        for cpart in cname:
            if kwargs[cpart] not in self.ids[ cpart ]:
                return 0
            my_id.append( self.ids[ cpart ][ kwargs[cpart] ] )
        my_id = [ tuple(my_id) ]
        
        return self.counts[ cname ][ tuple(zip(*my_id)) ][0]
    
    def items(self, typ):
        return sorted(self.ids[typ])
    
    def delete(self, typ, which):        
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
        
    def prune_zeros(self):
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
                
    def save_counts(self, name):
        import pickle

        with open(f'{name}.counts.pickle', 'wb') as outf:
            pickle.dump(self.counts, outf)

        with open(f'{name}.ids.pickle', 'wb') as outf:
            pickle.dump(self.ids, outf)
            
    def summarize(self):
        print( [(k, c.shape) for k,c in self.counts.items()])