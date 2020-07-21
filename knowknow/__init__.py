from IPython.core.display import display, HTML, Markdown, Image
from collections import Counter, defaultdict
from random import sample, shuffle
from itertools import chain

from tabulate import tabulate
import re

from pathlib import Path
from os.path import dirname, join
BASEDIR = dirname(__file__)

variable_dir = Path(BASEDIR,"variables")
#variable_dir = Path("C:\\Users\\amcga\\knowknow_variables")

from csv import reader as creader

import yaml
DOCS = yaml.load(
    Path(BASEDIR,"documentation.yaml").open('r',encoding='utf8'), 
    Loader=yaml.FullLoader
)

import networkx as nx
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

__all__ = [
    # constants
    "BASEDIR", "DOCS",
    "register_notebook",

    # getting counts
    "get_cnt", "save_cnt", "cnt_collapse",
    "get_cnt_keys", "plot_count_series",

    # plotting
    "plot_companions", "plot_context",
    'html_companions','partner_citations',
    'citation_characterize',

    # custom utils
    "canonical_citation",# "HumanName",

    'load_variable', 'save_variable',
    'save_figure', 'save_table',
    "VariableNotFound",
    
    "comb", "make_cross",
    
    "showdocs", "display_figure",
    "comments",

    # common imports
    "Counter", "defaultdict", "Path",
    "np", "nx","sns","pd", "re",
    "plt","display","HTML","Counter","Markdown","Image",
    'sample','shuffle','chain',
    'tabulate',"creader"
]

DEFAULT_KEYS = ['fj.fy','fy','c','c.fy']

def get_cnt_keys(name):
    avail = variable_dir.glob("%s ___ *"%name)
    avail = [x.name for x in avail]
    avail = [x.split("___")[1].strip() for x in avail]
    return avail




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





def comb(x,y):
    a = set(x.split("."))
    b = set(y.split("."))
    
    return ".".join(sorted(a.union(b)))

class comb_cont:
    def __init__(**kwargs):
        from collections import namedtuple

        ntkeys = sorted(k.split("."))
        my_named_tuple = namedtuple(k.replace(".","_"), ntkeys)
        return my_named_tuple(**dict(zip(ntkeys, x)))


cnt_cache = {}
def get_cnt(name, keys=None):
    
    if keys is None:
        keys = DEFAULT_KEYS

    cnt = {}

    for k in keys:
        if (name,k) in cnt_cache:
            cnt[k] = cnt_cache[(name,k)]
        else:
            varname = "%s ___ %s" % (name,k)

            #print(k)
            this_cnt = defaultdict(int, named_tupelize( dict(load_variable(varname)), k ))
            cnt[k] = this_cnt
            cnt_cache[(name,k)] = this_cnt

    avail = get_cnt_keys(name)

    print("Loaded keys: %s" % cnt.keys())
    print("Available keys: %s" % avail)
    return cnt

def save_cnt(name, data={}):
    
    for k,count in data.items():
        varname = "%s ___ %s" % (name,k)
        
        print("Saving %s" % varname)
        save_variable(varname, dict(count))

def cnt_collapse(c, f=lambda x:True, on=None):
    new_c = defaultdict(int)
    for item, count in c.items():
        if not f(item):
            continue

        new_c[on(item)] += count

    return Counter( dict(new_c.items()) )





class VariableNotFound(Exception):
    pass

data_files = {
    'sociology-wos': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/5eded795c67d30014e1f3714/?zip='
}

def download_file(url, outfn):
    import requests
    url = str(url)
    outfn = str(outfn)
    Path(outfn).parent.mkdir(exist_ok=True)
    print("Beginning download, ", url)
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(outfn, 'wb') as f:
            for i,chunk in enumerate(r.iter_content(chunk_size=8192)):
                if i % 1000 == 0 and i:
                    print('%0.2f MB downloaded...' % (i*8192/1e6))
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return outfn




def load_variable(name):
    import pickle
    from collections import defaultdict, Counter

    nsp = name.split("/")
    if len(nsp) == 1: # fallback to old ways
        nsp = name.split(".")
        collection = nsp[0]
        varname = ".".join(nsp[1:])
        name = "/".join([collection,varname])
    elif len(nsp) == 2:
        collection, varname = nsp
    else:
        raise Exception("idk how to parse this... help")

    if not variable_dir.joinpath(collection).exists():
        print("collection", collection, "does not exist...")
        print("attempting to load from OSF")

        if collection not in data_files:
            raise Exception("no data file logged for '%s'"%collection)

        zip_dest = Path(BASEDIR, "variables", "%s.zip" % collection)
        if not zip_dest.exists():
            download_file(data_files[collection], zip_dest)

        print("Extracting...", str(zip_dest))
        import zipfile
        with zipfile.ZipFile(str(zip_dest), 'r') as zip_ref:
            zip_ref.extractall(str(zip_dest.parent.joinpath(collection)))

    try:
        return pickle.load( variable_dir.joinpath(name).open('rb') )
    except FileNotFoundError:
        raise VariableNotFound(name)

def save_variable(name, val):
    
    nsp = name.split("/")
    if len(nsp) == 1: # fallback to old ways
        nsp = name.split(".")
        collection = nsp[0]
        varname = ".".join(nsp[1:])
        name = "/".join([collection,varname])
    elif len(nsp) == 2:
        collection, varname = nsp
    else:
        raise Exception("idk how to parse this... help - e.g. sociology-wos/c.ysum")
    
    import pickle
    
    variable_dir.joinpath(name).parent.mkdir(exist_ok=True)
    
    pickle.dump( val, variable_dir.joinpath(name).open('wb') )







# ============================================================
# ============================================================
# ======================== UTILITIES =========================
# ============================================================
# ============================================================


if False:
    from nameparser import HumanName as HumanNameOld
    class HumanName(HumanNameOld):
        def __init__(self, *args, **kwargs):
            args = list(args)
            
            self._gender = None

            mustbe = ["Jr","Sr","Mr","Mrs"]
            for mb in mustbe:
                args[0] = re.sub("(%s)([^.a-zA-Z]|$)" % re.escape(mb), r"\1.\2", args[0])

            super(HumanName, self).__init__(*args, **kwargs)

        @property
        def gender(self):
            # generate the property if we don't have it already
            if self._gender is None:
                gendr = 'unknown'
                if self.first:
                    gendr = gender_detector.get_gender(self.first)
                if self.title:
                    if self.title.lower() in ['princess', 'ms.', 'mrs.', 'queen']:
                        gendr = 'female'
                    if self.title.lower() in ['mr.', 'prince', 'king']:
                        gendr = 'male'

                self._gender = gendr

            return self._gender

        def supercedes(self, compare):

            # if the last name is there, it must match
            if compare.last != self.last:
                return False

            # if the first name is there, it must match
            if compare.first and self.first:
                # handling initials
                if self.is_an_initial(compare.first) or self.is_an_initial(self.first):
                    if self.first[0].lower() != compare.first[0].lower():
                        return False
                else:
                    if self.first != compare.first:
                        return False

            # if the middle name is there, it must match
            if compare.middle and self.middle:
                # handling initials
                if self.is_an_initial(compare.middle) or self.is_an_initial(self.middle):
                    if self.middle[0].lower() != compare.middle[0].lower():
                        return False
                else:
                    if self.middle != compare.middle:
                        return False

            # make sure the genders match --
            # note -- only if there's no first name
            if not compare.first and compare.title:
                # Mr. Mrs.
                m = ["Mr."]
                f = ["Mrs.", "Ms."]

                if compare.title in m and self.gender == 'female':
                    print("Skipping", compare)
                    return False
                elif compare.title in f and self.gender == 'male':
                    print("Skipping", compare)
                    return False

            # also, Jr.'s are important!
            # only if there's a first name and last name
            if compare.first and compare.last:
                if compare.suffix and self.suffix:
                    if compare.suffix != self.suffix:
                        return False

            return True


def convert_individual(x):
    reg_pattern = r'^%s((,| and| et al).*)?$'

    if type(x) == dict:
            return x
    if type(x) == str:
        sp = x.split( " " )
        if len(sp)>1:
            return {
                "0":{"$regex": reg_pattern % " ".join(sp[:-1] ), "$options":'i'},
                "1": sp[-1]
            }

        sp = sp[0]
        try:
            return {
                "1": int(sp)
            }
        except:
            return {
                "0": {"$regex": reg_pattern % sp, "$options":'i'}
            }
    if type(x) == int:
        return {
            "1":x
        }
    if type(x) == tuple:
        if sum(y is not None for y in x) == 0:
            raise Exception("nothing in this individual search...")

        if x[0] is None:
            return {
                "1":x[1]
            }
        elif x[1] is None:
            return {
                "0":{"$regex": reg_pattern % x[0], "$options":'i'}
            }
        else:
            return {
                "0":{"$regex": reg_pattern % x[0], "$options":'i'},
                "1":x[1]
            }

    raise Exception("citation guy not dict,str,tuple, or int")





def partner_counter(cnt, who):
    who = canonical_citation(who)
    return Counter(dict(
        [ (x2,v) for (x1,x2), v in cnt['cc'].items() if v > 0 and x1 == who ]+
        [ (x1,v) for (x1,x2), v in cnt['cc'].items() if v > 0 and x2 == who ]
    ))

def canonical_citation(who):
    if type(who) == str:
        whos = who.split()
        if len(whos) < 2:
            raise Exception("Canonicalizing a citation requires it to *be* a citation...")
        who = (" ".join(whos[:-1]), whos[-1])
    who = tuple(who)
    return who


def partner_citations(*authors):

    searchTerms = [
        {"citations":{"$elemMatch": convert_individual(author)}}
        for author in authors
    ]
    #print(searchTerms)

    search = {"$and": searchTerms }

    return CIT_DB.find(search)












# ============================================================
# ============================================================
# ======================== SUMMARY FUNCTIONS =================
# ============================================================
# ============================================================






def plot_companions(cnt, who):
    # time trends of this work with others it is cited with
    who = canonical_citation(who)
    others = partner_counter(cnt, who)

    to_track = [who] + [tuple(x[0]) for x in others.most_common(7)]
    #print(to_track)
    years = list(range(1970, 2020))

    trends = {c:[ cnt['c.fy'][(c,y)] for y in years ] for c in to_track}
    #for c in trends:
    #    plt.plot( years, trends[c], label=c )
    from colour import Color

    A = Color("#464159")
    B = Color("#c7f0db")
    C = Color("#6c7b95")

    plt.figure(figsize=(15,5))
    #print([ x.hex for x in A.range_to(B,len(trends)) ])
    plt.stackplot(
        years,
        *list(trends.values()),
        labels=[" ".join(x) for x in trends.keys()],
        colors=[ x.hex for x in A.range_to(B,len(trends)) ]
    )
    plt.legend()



def plot_context(context, draw_totals=False, n=15):

    cits = list( CIT_DB.find({"contextPure":{"$regex":r".*%s.*" % context}}) )
    print(len(cits), "citations found")

    for c in cits:
        c['doc'] = {
            k:v
            for k,v in
                THEORY_DB.find_one({"_id":c['doc']}).items()
            if k in ['_id', 'year', 'authors', 'title', 'journal']
        }

    cnt = get_cnt('doc')

    #total = np.array([cnt['y'][y] for y in range(1850,2020)])
    #print("%s years" % len(total))
    #plt.plot(range(1850,2020), total)

    from collections import Counter
    all_years = [c['doc']['year'] for c in cits]
    yearc = Counter(all_years)

    yrs = np.arange( min(all_years), max(all_years)+1 )

    yt = np.array([cnt['y'][y] for y in yrs])
    c = [ yearc[k] for k in yrs ]

    # plot the year, divided by the total # docs for that year...
    plt.plot(yrs, c/yt)

    # term counts
    if False:
        ayc = defaultdict(lambda:defaultdict(int))
        ac = defaultdict(int)
        for co in cits:
            for citation in co['citations']:
                ac[tuple(citation)] += 1
                ayc[tuple(citation)][co['doc']['year']] += 1

    ayc = defaultdict(lambda:defaultdict(set))
    ac = defaultdict(set)
    for co in cits:
        for citation in co['citations']:
            ac[tuple(citation)].add(co['doc']['_id'])
            ayc[tuple(citation)][co['doc']['year']].add(co['doc']['_id'])
    ayc = { a: {
        y: len(ayc[a][y])
        for y in yrs
    } for a in ayc }
    ac = {
        a: len(ac[a])
        for a in ac
    }

    to_plot = [x[0] for x in Counter(dict(ac.items())).most_common(n)]

    year_counts = {a: ayc[a] for a in to_plot}
    year_min = min( min(v for v,c in yc.items() if c > 0) for yc in year_counts.values() )
    year_max = max( max(v for v,c in yc.items() if c > 0) for yc in year_counts.values() )

    years = list(range(year_min, year_max+1 -1)) #we subtract one because the last year (2015) is incomplete

    year_counts = {
        a: np.array( list( ys[yi] for yi in years ) )
        for a, ys in year_counts.items()
    }

    from colour import Color

    A = Color("#464159")
    B = Color("#c7f0db")
    C = Color("#6c7b95")

    everything = np.array([
        [
            ayc[a][y]
            for y in years
        ]
        for a in ayc.keys()
    ])

    doc_totals = np.sum( everything > 0, axis=0 )
    totals = np.sum( everything, axis=0 )

    hs = []

    fig, ax = plt.subplots(figsize=(20,10))

    yt = np.array([cnt['y'][y] for y in years])

    h = plt.stackplot(
        years,
        *[yci for yci in year_counts.values()],
        labels=[" ".join(x) for x in year_counts.keys()],
        colors=[ x.hex for x in A.range_to(B,n) ]
    )
    h = h[0]

    #for k in year_counts:
    #    h, = plt.plot(years, year_counts[k], label=" ".join(k))
    #    hs.append(h)
    #plt.legend(handles=hs)

    if draw_totals:
        plt.plot(
            years,
            doc_totals,
            color="#000000"
        )

        plt.plot(
            years,
            totals,
            color="#000000",
            dashes=(2,1)
        )

    ax.legend(loc='upper left')
    plt.show()


    return cits







def html_companions(query, outter=5, inner=5):
    query = canonical_citation(query)
    # just a table

    others = partner_counter(query)
    #print(others.most_common(10))

    html = []

    num_printed_outter = 0
    for item, c in others.most_common(outter):
        if item == query:
            continue
        if c == 1:
            continue

        html.append( "<b>%s</b>" % "(%s) %s" % (c, "%s (%s)" % item) )

        html.append("<ul>")
        num_printed = 0

        us_together = list(partner_citations(query, item))
        for r in sample(us_together, min(inner, len(us_together))):
            html.append("<li>%s</li>" % r['contextPure'])

        html.append("</ul>")

        num_printed_outter += 1
        if num_printed_outter >= outter:
            break

    html = "\n".join(html)
    return html


def citation_characterize(
    query,
    eliminate_substrings = True,
    skip_identical_p = True,
    min_count = 5,
    max_tuple = 5,
    max_pv = 0.001,

    max_to_print_outter = 10,
    max_to_print_inner = 3
):

    query = canonical_citation(query)
    results = list(partner_citations(query))

    from collections import Counter, defaultdict

    from sklearn.feature_selection import chi2
    from sklearn.feature_extraction.text import CountVectorizer








    #
    # retrieve all citations from the docs the focal results came from
    #

    docs = set( cz['doc'] for cz in results )
    all_citations = list( CIT_DB.find({"doc": {"$in": list(docs)}}) )

    #
    # count the words in each of the citations into a huge sparse array
    #

    cv = CountVectorizer( ngram_range=(1,max_tuple+1) )
    X = cv.fit_transform( x['contextPure'] for x in all_citations )





    html = []

    #
    # compute chi-2 value predicting whether the ngram is next to this or that citation
    #

    Y = np.array([
        list(query) in x['citations']
        for x in all_citations
    ])

    ch, pv = chi2(X, Y)

    #
    # decide the most important words
    #

    raw_word_counts = np.array(X[Y,:].sum(axis=0)).flatten()
    word_indices = np.argwhere((pv <= max_pv) & (raw_word_counts >= min_count))
    print(len(word_indices), 'words with these restrictions')

    #
    # look up these words in the feature dictionary and print
    #

    features = np.array( cv.get_feature_names() )
    feature2id = { f:i for i,f in enumerate(features) }

    topr = features[ word_indices ]
    topr = [x[0] for x in topr]

    def not_smaller(x):
        for y in topr:
            if x in y and x != y:
                return False
        return True

    already_seen = []

    prev_p = None
    nskipped = 0

    num_printed = 0
    for x in sorted(topr, key=lambda x: pv[ feature2id[x] ]):
        inside=False
        for y in already_seen:
            if x in y or y in x:
                inside=True
        if inside and eliminate_substrings:
            continue
        already_seen.append(x)

        myp = pv[ feature2id[x] ]
        if myp == prev_p and skip_identical_p:
            nskipped += 1
            continue

        if nskipped > 0:
            html.append("<p>%s (identical p-value)</p>" % (str(nskipped) + " skipped"))
            nskipped = 0

        prev_p = myp


        num_inner = 0
        todo = []
        for c in all_citations:
            if list(query) not in c['citations']:
                continue
            if x.lower() not in c['contextPure'].lower():
                continue

            todo.append(c)
        shuffle(todo)


        html.append("<b>%s</b> (-log(p)=%0.2f, N=%s)"%(x, -np.log(myp), len(todo)))




        html.append("<ul>")

        for c in todo:
            mydoc = THEORY_DB.find_one({"_id":c['doc']})

            html.append("<li>%s (%s)</li>" % (
                c['contextPure'].replace(x, "<b>%s</b>" % x),
                "<i>from %s</i>" % "%s, %s; %s" % (
                    ", ".join( an.split()[-1] for an in mydoc['authors'] ),
                    mydoc['year'],
                    mydoc['journal']
                )
            ))

            num_inner += 1
            if num_inner >= max_to_print_inner:
                break

        html.append("</ul>")


        num_printed += 1
        if num_printed >= max_to_print_outter:
            break

    todo = sample(all_citations, 10)
    html.append("<ul>")

    for c in todo:
        mydoc = THEORY_DB.find_one({"_id":c['doc']})

        html.append("<li>%s (%s)</li>" % (
            c['contextPure'],
            "<i>from %s</i>" % "%s, %s; %s" % (
                ", ".join( an.split()[-1] for an in mydoc['authors'] ),
                mydoc['year'],
                mydoc['journal']
            )
        ))
    html.append("</ul>")

    html = "\n".join(html)
    return html

#NB_FNAME = None
NB_DIR = None
def register_notebook(fname, title):
    global NB_FNAME, NB_DIR
    #NB_FNAME = fname
    NB_DIR = Path(fname)

def save_figure(name):
    global NB_FNAME, NB_DIR
    outdir = NB_DIR.joinpath("figures")
    if not outdir.exists():
        outdir.mkdir()
    print("Saving to '%s'"%outdir)
    plt.savefig(str(outdir.joinpath("%s.png" % name)), bbox_inches="tight")



"""
#the pandas way:

df = pd.DataFrame.from_records(rows)
df.columns = headers
df.to_latex(
    index_names=False,
    column_format = "|%s|" % "|".join( "p{%scm}"%(i+1) for i in range(len(df.columns)) )
)
"""

def latex_escape(x):
    if type(x) == int:
        x = '{:,}'.format(x)
    else:
        x = str(x)

    x = x.replace("%", "\\%")
    x = x.replace("&", "\\&")
    return x

def header_generator(hs):
    from itertools import groupby
    for h, g in groupby(hs):
        h = latex_escape(h)
        n = len(list(g))
        if n > 1:
            yield "\multicolumn{%s}{c|}{%s}" % (n,h)
        else:
            yield h


def save_table_latex(name, *args, **kwargs):
    outdir = Path(BASEDIR,"tables")
    if not outdir.exists():
        outdir.mkdir()
    outfn = str(outdir.joinpath("%s.tex" % name))

    tab_latex = create_table(*args, **kwargs)

    with open(outfn, 'w', encoding='utf8') as outf:
        outf.write(tab_latex)

def save_table_html(html, name):
    outdir = Path(BASEDIR,"tables")
    if not outdir.exists():
        outdir.mkdir()
    outfn = str(outdir.joinpath("%s.html" % name))
    with open(outfn, 'w', encoding='utf8') as outf:
        outf.write(html)
        
def save_table(name, *args, **kwargs):
    if 'html' in kwargs:
        save_table_html(name, kwargs['html'])
        
def create_table(rows, headers, caption='', footnotes='', widths=None, columns=2):
    assert(type(headers) == list)
    if not type(headers[0]) == list:
        headers = [headers]
    assert(len(rows[0]) == len(headers[0]))
    assert(columns in {1,2})

    rows = [[latex_escape(x) for x in r] for r in rows]

    caption, footnotes = tuple(map(latex_escape, [caption, footnotes]))

    columndef = "|%s|" % "|".join("p{%scm}"%x for x in widths)
    col2star = "*" if columns==2 else ""

    print([
        list(header_generator(h))
        for h in headers
    ])

    header = "\n".join(
        "%s \\\\" % "&".join(header_generator(h))
        for h in headers
    )
    content = "\n".join(
        "%s \\\\" % "&".join(r)
        for r in rows
    )
    tab = """
\\begin{table%s}[!htb]
\\caption{%s}
\\begin{tabular}{%s}
\\tophline
%s
\\middlehline
%s
\\bottomhline
\\end{tabular}
\\belowtable{%s}
\\end{table%s}
""" % (col2star, caption, columndef, header, content, footnotes, col2star)
    return tab




def plot_count_series(alltop, database, myname=None, 
    overwrite=True, 
    markers={}, print_names=None, 
    ctype='c', yearly_prop=False, 
    xlim_display=(1890,2030), xlim_data=(1900,2020),
    cols = 5,
    rows_per_group = 2,
        count_unit='doc', label_num='max'):
    
    doing_both = False
    if type(count_unit) != str:
        if set(count_unit) == {'doc','ind'}:
            cits = get_cnt("%s.doc" % (database), [comb(ctype,'fy'),'fy'])
            ind = get_cnt("%s.ind" % (database), [comb(ctype,'fy'),'fy'])
            doing_both = True
        else:
            raise Exception('whoopsie')
    else:
        cits = get_cnt("%s.%s" % (database,count_unit), [comb(ctype,'fy'),'fy'])

    rows = len(alltop) // cols + bool(len(alltop) % cols) # = 15 for 5
    
    groupsize = rows_per_group * cols
    gs = rows // rows_per_group + bool(rows % rows_per_group)

    for groupi in range(gs):
        myFnName = "%s.%s" % (myname,groupi)
        if Path(myFnName).exists() and not overwrite:
            continue
        
        plt.figure(figsize=(cols*4,rows*1.2))
        plt.subplots_adjust(hspace=0.6, wspace=0)

        for i,name in enumerate(alltop[ groupi*groupsize : (groupi+1)*groupsize]):
            print(name)
            
            plt.subplot(rows,cols,i+1)

            yrstart,yrend = xlim_data
            years = range(yrstart, yrend)
            vals = [cits[ comb('fy' , ctype) ][
                make_cross({
                    'fy': y,
                    ctype: name
                })
            ] for y in years]
            
            if yearly_prop:
                vals = [
                    vals[yi] / cits['fy'][(y,)]
                    if vals[yi]>0 else 0
                    for yi,y in enumerate(years)
                ]

            plt.fill_between(years,vals,color='black',alpha=0.4)
            if doing_both:
                v2 = np.array([ind[ comb('fy' , ctype) ][
                        make_cross({
                            'fy': y,
                            ctype: name
                        })
                    ] for y in years])
                v2 = max(vals) * v2 * 0.5 / np.max(v2)
                plt.plot(years, v2, color='black')
            
            if print_names is not None:
                title = print_names(name)
            else:
                title = name
                
            if len(title) >= 40:
                title = title[:37]+"..."
                
            #t = plt.text(min(years) + (max(years)-min(years))/2, 1.2*(max(vals))/1,title, fontsize=13)
            plt.title(title)
            
            plt.axis('off')

            lines = []
            for decade in range(yrstart,yrend,10):
                lines += [
                    (decade+1, decade+10-1), 
                    (-max(vals)/5, -max(vals)/5), 
                    'black'
                ]

            # labeling the last
            if label_num == 'last':
                last_num = vals[-1]

                if yearly_prop:
                    last_num = "%0.1f%%"%(last_num*100)

                lines += [
                    (xlim_display[1]-10, xlim_display[1]-5),
                    (last_num, last_num),
                    "black"
                ]
                plt.text(xlim_display[1]-3, last_num, last_num, fontsize=12, verticalalignment='center', horizontalalignment='left')#*3+min(vals)
                
            elif label_num == 'max':
                max_num = max(vals)

                if yearly_prop:
                    max_num = "%0.1f%%"%(max_num*100)

                lines += [
                    (xlim_display[1]-10, xlim_display[1]-5),
                    (max_num, max_num),
                    "black"
                ]
                plt.text(xlim_display[1]-3, max_num, max_num, fontsize=12, verticalalignment='center', horizontalalignment='left')#*3+min(vals)
            elif label_num is not None:
                raise Exception('don\'t recognize that label_num...')

            plt.plot(*lines)
            plt.scatter(
                [1900,1950,2000],
                [-max(vals)/5]*3,
                color='black',
                s=20
            )
            
            plt.xlim(xlim_display)
            
            if name in markers:
                ms = markers[ name ]
                if type(ms) == list:
                    plt.scatter(
                        ms,
                        [-max(vals)/2]*len(ms),
                        color='red',
                        s=30,
                        marker="^"
                    )
                elif type(ms) == dict:
                    for k,v in ms.items():
                        plt.text(k, -max(vals)/2, v)

        save_figure(myFnName)
        plt.show();
        
        
        
        
        
# print the documentation for this file, if it exists!
def showdocs(uid, title=""):
    
    if uid not in DOCS:
        display(Markdown("No documentation found for `%s`"%uid))
        return
    
    d = DOCS[uid]

    if len(title):
        title += " "
    
    display(Markdown("# %s%s" % (title, d['name'])))
    display(Markdown(d['desc']))
    
    if 'fn' in d:
        # display the figures.
        files = Path(BASEDIR, 'figures').glob("%s.png" % d['fn'])
        for f in sorted(files,key=lambda x:x.name):
            #print(f.name)
            display(Image(filename=f))    
    
    # citations
    md = []
    if 'refs' in d and len(d['refs']):
        md += ["# References"]
        md += ["\n".join("+ %s" % z for z in d['refs'])]
    
    display(Markdown("\n\n\n\n".join(md)))
    
    
    
    

def display_figure(x, titles=True):
    displayed_normally = set()

    import os
    import time
    from datetime import datetime
    
    if x in DOCS:

        d = DOCS[x]

        if titles:
            display(HTML("<h1 id='%s'>%s</h1>" % (x, d['name'])))
        display(Markdown(d['desc']))

        if 'refs' in d:
            display(Markdown("\n".join("+ %s"%x for x in d['refs'])))

        files = Path(NB_DIR, 'figures').glob("%s.png" % d['fn'])
        for f in sorted(files,key=lambda x:x.name):
            #print(f.name)
            display(Image(filename=f))
            displayed_normally.add(f.name)
            
    else:
        
        search = list(Path(NB_DIR, 'figures').glob(x))
        if not len(search):
            raise Exception("What figure!? '%s'" % x)
        
        for f in sorted(search):
            if titles:
                display(HTML("<h1>%s</h1>" % (f.name)))
            display(Image(filename=f))
            displayed_normally.add(f.name)
            
    return displayed_normally
            
            
def comments():
    html = """<script src="https://unpkg.com/commentbox.io/dist/commentBox.min.js"></script><div class="commentbox"></div><script type="module">
    import commentBox from 'commentbox.io';commentBox('5738033810767872-proj');
    </script>
    """

    display(HTML(html))