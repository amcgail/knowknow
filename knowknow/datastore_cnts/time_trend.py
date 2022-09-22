from .. import *
from .count_cache import Dataset, make_cross
from ..exceptions import invalid_entry
from . import stats


__all__ = [
    'ttDF', 'TimeTrend'
]

class ttDF:

    def __init__(self, dataset_name, dtype, debug=False):
        self.meta = defaultdict(int)
        self.debug = debug

        self.dataset_name = dataset_name
        self.dataset = Dataset(dataset_name)
        self.dtype = dtype
        self.final = {}

    def run(self):
        self.generate_timetrends()
        self.persist()

    def persist(self):
        varname = "%s.ysum" % (self.dtype)
        self.dataset.save_variable(varname, self.final)
        print('success!')

    def generate_timetrends(self):
        import tracemalloc

        tracemalloc.start()

        items = list(self.dataset.items(self.dtype))

        for ci, item in enumerate(items):

            if ci % 50000 == 0:
                print("Item %s/%s (%s)" % (ci, len(items), item))
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                print(self.meta)

            if self.debug and ci >= 50000:
                break

            dp = self.make_tt_dump( item )
            if dp is None:
                continue
            self.final[item] = dp

        tracemalloc.stop()

    def make_tt_dump(self, item):
        # this function has an outstanding memory leak!
        self.meta['at least one citation'] += 1

        if self.dataset['type'] == 'wos' and self.dtype == 'c':
            sp = item.split("|")
            if len(sp) < 2:
                self.meta['not enough parts'] += 1
                #raise Exception('wut', item)
                return None

        # memory-inexpensive check...
        if self.dataset(**{self.dtype: item}).docs < 5:
            self.meta['less than 5 citations... dropped.2'] += 1
            return None

        # create a timetrend
        try:
            tt = TimeTrend(name=item, dtype=self.dtype, dataset=self.dataset)
        except invalid_entry:
            self.meta['invalid entry'] += 1
            return None

        # small error catch
        if hasattr(tt, 'pub') and tt.first < tt.pub:
            self.meta['first citation before pub date'] += 1
            return None

        # don't care about those with only a single citation
        if tt.total < 5:
            self.meta['less than 5 citations... dropped.'] += 1
            return None

        # we really don't care about those that never rise in use
        #if tt.first == tt.maxpropy:
        #    self.meta['never rise'] += 1
        #    continue

        self.meta['passed tests pre-blacklist'] += 1

        # set some attributes before it's persisted

        dp = tt.dump()
        return dp




class TimeTrend:
    
    def __init__(self, dataset, dtype, name, years=None):
        self.dataset = dataset
        self.dtype = dtype

        if years is not None:
            self.data_start, self.data_end = years
        elif 'data_start' in self.dataset and 'data_end' in self.dataset:
            self.data_start, self.data_end = self.dataset['data_start'], self.dataset['data_end']
        else:
            raise Exception("dataset has no `data_start` and `data_end` variables.\nset these or the years= parameter")
        
        self._cumsum = None
        self._c = None

        # this marks the end of adding non-saving variables to the class
        self.dict_filter = set(list(self.__dict__) + ["dict_filter"])  # save dict keys

        if type(name) == tuple:
            self.name = name[0]
        else:
            self.name = name

        self.simple_stats()

    def dump(self):
        to_return_keys = set(self.__dict__).difference(self.dict_filter)
        
        return {
            k: getattr(self,k)
            for k in to_return_keys
            if k[0] != '_'
        }
    
    @classmethod
    def addinit(cls, st):
        if st not in cls.init_stack:
            cls.init_stack.append(st)

    def series(self, A=None, B=None):
        if A is None:
            A = self.data_start
        if B is None:
            B = self.data_end
        return [self.c[YY] for YY in range(A, B+1)]

    def show(self):
        plt.plot(
            range(self.data_start, self.data_end+1),
            self.cits(self.data_start, self.data_end),
            label=self.name
        )

    def load_c(self):
        # these counts act as the data for all subsequent computations
        all_by_y = defaultdict( int, self.dataset.items('fy') )

        self._c = defaultdict(lambda:None)
        self._cp = defaultdict(lambda:None)

        # grab the counts!
        for YY in range( 
            self.data_start,
                self.data_end + 1
        ):
            cc = self.dataset(**{self.dtype:self.name, 'fy':YY})
            self._c[YY] = cc
            self._cp[YY] = cc / all_by_y[YY] if all_by_y[YY] > 0 else 0

    def old_load_d(self):
        # these counts act as the data for all subsequent computations
        my_by_y = self.dataset.by(self.dtype, 'fy').docs
        all_by_y = self.dataset.by('fy').docs

        self._d = defaultdict(lambda:None)
        self._dp = defaultdict(lambda:None)

        for YY in range( 
            self.data_start,
                self.data_end + 1
        ):
            dy = make_cross(**{self.dtype:self.name, 'fy':YY})
            self._d[YY] = my_by_y[dy]
            self._dp[YY] = my_by_y[dy] / all_by_y[(YY,)] if all_by_y[(YY,)] > 0 else 0

    @property
    def c(self):
        if self._c is None:
            self.load_c()

        return self._c
    @property
    def cp(self):
        if self._cp is None:
            self.load_c()

        return self._cp

    if False:
        @property
        def d(self):
            if self._d is None:
                self.load_d()
                
            return self._d
        @property
        def dp(self):
            if self._dp is None:
                self.load_d()
                
            return self._dp
            


    def simple_stats(self):

        non_zero_years = [k for k,c in self.c.items() if c>0]
        if not len(non_zero_years):
            raise invalid_entry('no with positive count', self.name)

        self.first = min(non_zero_years)
        self.last = max(non_zero_years)

        self.maxcounty = max(self.c, key=lambda y:(self.c[y],y))
        self.maxpropy = max(self.cp, key=lambda y:(self.cp[y],y))

        self.maxprop = self.cp[ self.maxpropy ]
        self.maxcount = self.c[ self.maxcounty ]

        self.total = sum(self.c.values())
        self.totalprop = sum(self.cp.values())
        
        if self.dtype == 'c':
            # extracts some extra information from the name

            if 'type' in self.dataset:
                
                self.type = 'article'
                if self.dataset['type'] == 'wos':
                    sp = self.name.split("|")

                    if not len(sp):
                        print('Wtf',sp)
                        raise

                    try:
                        self.pub = int(sp[1])
                        self.type = 'article'
                    except ValueError:
                        self.type = 'book'
                        #self.pub = pubyears[self.name]

                elif self.dataset['type'] == 'jstor':
                    inparens = re.findall(r'\(([^)]+)\)', self.name)[0]
                    self.pub = int(inparens)

    def avg_between(self, A, B): # including B
        return self.sum_between(A, B) / (B-A+1)
            
    def sum_between(self, A, B): # including B
        if self._cumsum is None:
            # cumsum makes sum_between's very efficient!
            self._cumsum = [self.c[self.data_start]]
            for YY in range(
                    self.data_start + 1,
                    self.data_end + 1
                ):
                self._cumsum.append( self._cumsum[-1] + self.c[YY] )
            self._cumsum = np.array(self._cumsum)

        if A < self.data_start:
            raise Exception("data begins at %s. cannot get %s" % (self.data_start, A))
        if B > self.data_end:
            raise Exception("data ends at %s. cannot get %s" % (self.data_end, B))

        if not (A <= B):
            raise Exception("Can only sum forwards in time. Fix your code...")

        if A == self.data_start:
            return self._cumsum[B - self.data_start]
        else:
            return self._cumsum[B - self.data_start] - self._cumsum[(A - 1) - self.data_start]


    def births_deaths(self, **kwargs):
        """
        def births_deaths(tt,  # the trend this should be calculated on
                          birth_cutoff=0.5,  # relative to the last life
                          death_cutoff=0.1,  # relative to this life so far
                          forward_view=10,  # years to look forward for comparison
                          backward_view=5  # minimum years to look back for comparison
                          ):
        """
        return stats.births_deaths(self, **kwargs)

    def __iter__(self):
        return ( self.c[YY] for YY in range(self.data_start, self.data_end+1) )
