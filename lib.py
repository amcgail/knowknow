from IPython.core.display import display, HTML
from collections import Counter, defaultdict
from random import sample, shuffle
from itertools import chain

from tabulate import tabulate


from pathlib import Path
from os.path import dirname, join
BASEDIR = dirname(__file__)

import networkx as nx
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

__all__ = [
    # constants
    "BASEDIR",
    
    # database hooks
    "CIT_DB", "THEORY_DB", "BIB_DB", "WIKI_DB",
    
    # getting counts
    "get_cnt", "cnt_collapse",
    
    # plotting
    "plot_companions", "plot_context",
    'html_companions','partner_citations',
    'citation_characterize',
    
    # custom utils
    "canonical_citation",# "HumanName",
    
    'load_variable', 'save_variable',
    'save_figure', 'save_table',
    
    # common imports
    "Counter", "defaultdict", "Path", 
    "np", "nx","sns","pd",
    "plt","display","HTML","Counter",
    'sample','shuffle','chain',
    'tabulate'
]


# DB
from pymongo import MongoClient

db = MongoClient()['journal_articles']
CIT_DB = db['citations']
THEORY_DB = db['theory']
WIKI_DB = db['wiki_sociologists']
BIB_DB = db['bibliography']

cnt_cache = {}
def get_cnt(name):
    if name in cnt_cache:
        cnt = cnt_cache[name]
    else:
        names = {
            "intext.doc": '../120 _ count co-occurrences/cnt_doc.pickle',
            "intext.ind": '../120 _ count co-occurrences/cnt_ind.pickle',
            "doc": '../120 _ count co-occurrences/THEORY_DB/cnt_doc.pickle',
            'wos.all.uid.doc': '../120 _ count co-occurrences/wos_counter_larger/cnt_doc_fullUnique.pickle',
            'wos.all.uid.ind': '../120 _ count co-occurrences/wos_counter_larger/cnt_ind_fullUnique.pickle',
        }
        if name not in names:
            raise Exception("Please choose one of: %s" % "; ".join(names))

        print("loading from %s"%names[name])

        # load all the counts
        import pickle
        from collections import defaultdict, Counter

        cnt = pickle.load( open( join(BASEDIR,names[name]), 'rb') )
        for k in cnt:
            cnt[k] = defaultdict(int, cnt[k])

        cnt_cache[name] = cnt
        
    print(cnt.keys())
    return cnt

def cnt_collapse(c, f=lambda x:True, on=None):
    new_c = defaultdict(int)
    for item, count in c.items():
        if not f(item):
            continue
        
        new_c[on(item)] += count
        
    return Counter( dict(new_c.items()) )







def load_variable(name):
    import pickle
    from collections import defaultdict, Counter
    
    return pickle.load( open( join(BASEDIR,"variables",name), 'rb') )

def save_variable(name, val):
    import pickle
    pickle.dump( val, open( join(BASEDIR,"variables",name), 'wb') )
    
            





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

            import re
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

    trends = {c:[ cnt['cy'][(c,y)] for y in years ] for c in to_track}
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

def save_figure(name):
    outdir = Path(BASEDIR,"figures")
    if not outdir.exists():
        outdir.mkdir()
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


def save_table(name, *args, **kwargs):
    outdir = Path(BASEDIR,"tables")
    if not outdir.exists():
        outdir.mkdir()
    outfn = str(outdir.joinpath("%s.tex" % name))
    
    tab_latex = create_table(*args, **kwargs)
    
    with open(outfn, 'w', encoding='utf8') as outf:
        outf.write(tab_latex)

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
