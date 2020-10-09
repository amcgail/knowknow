from .dbmodel import Doc,Cit,Keyword,db
from peewee import *
from collections import Counter

from pathlib import Path
from hashlib import md5
import re




__all__ = [
    'FILT', 'BY', 'UTIL',
    'count_docs', 'count_cits',

    'biggest_docs', 'biggest_cits',
    'Cit1', 'Cit2',
    'bib_network',

    'Doc','Cit'
]






class FILT:pass
FILT.big3america = [
    (Doc.journal, [
        'american journal of sociology',
        'american review of sociology',
        "social forces"
    ])
]
FILT.none = []



class BY:pass
BY.year_published = {'y':Doc.year}


class UTIL:pass
UTIL.count_docs = fn.COUNT(Doc.id.distinct()).alias("count")
UTIL.count_cits = fn.COUNT().alias("count")












def filt_it( query, filt ):
    
    for thing in filt:
        if type(thing) == tuple:
            f, where_what = thing
            if type(where_what) == list:
                query = query.where(f.in_(where_what))
            elif type(where_what) == str:
                if where_what[-1] == "*":
                    query = query.where(f.contains(where_what[:-1]))
                else:
                    query = query.where(f == where_what)   
            else:
                raise Exception('what to do with this type?', type(where_what))
        elif type(thing) == Expression:
            query = query.where(thing)
        else:
            raise Exception('what to do with this type?', type(thing))

    return query

def getq_dc( filt=[] ):
    from peewee import Expression
    query = (Doc
        .select()
        .join(Cit, on=(Cit.doc==Doc.id))
    )

    query = filt_it( query, filt )
    return query


def count_docs(**kwargs):
    kwargs['count'] = UTIL.count_docs
    return count_table(**kwargs)
def count_cits(**kwargs):
    kwargs['count'] = UTIL.count_cits
    return count_table(**kwargs)

def count_table( 
    filt=[], #+FILT.big3america #+ [(Cit.author,'Bourdieu, P.')]
    by=BY.year_published,
    start=1950,
    end=2020,
    count=UTIL.count_docs,
    min_count = -1
):


    from collections import defaultdict
    import pandas as pd

    query = getq_dc(filt)

    query = query.select(
        count, 
        *[ v.alias(k) for k,v in by.items()]
    )

    query = (query
        .group_by(*list(by.values()))
        .order_by(*list(by.values()))
    )

    if min_count != -1:
        query = query.having(SQL('count') >= min_count)

    #print(query.sql())
    #print(query.count())

    # do the query
    query = list(query)

    def produce_rows(query):
        for r in query:
            # retrieve the entries from the SQL query
            outr = {}
            for k,v in by.items():
                #print(k,v)
                if v.model.__name__ == 'Doc':
                    outr[k] = getattr(r, k)
                elif v.model.__name__ == 'Cit':
                    outr[k] = getattr(r.cit, k)

            outr['count'] = r.count
            #outr['doc_id'] = r.id
                
            yield outr

    # convert query to DF
    dd = pd.DataFrame(
        list(produce_rows(query))
    )

    #dd = dd.set_index(
    #    list(by.keys())
    #)

    return dd


def get_model_from_row(row, what):
    if what.model.__name__ == 'Cit':
        return getattr(row.cit, what)
    if what.model.__name__ == 'Doc':
        return getattr(row, what)

def biggest_docs(filt=[], what=Doc.first_author):
    from collections import defaultdict
    q = (getq_dc(filt)
        .select(what.alias('what'), UTIL.count_docs)
        .group_by(what)
        .order_by(SQL("count").desc())
        .limit(100)
    )
    
    def jj(x):
        if what.model.__name__ == 'Cit':
            return x.cit.what
        if what.model.__name__ == 'Doc':
            return x.what

    return [
        ( jj(d), d.count)
        for d in q
    ]

def biggest_cits(filt=[], what=Cit.author, N=100):
    from collections import defaultdict

    q = (getq_dc(filt)
        .select(what.alias('what'), UTIL.count_cits)
        .group_by(what)
        .order_by(SQL("count").desc())
        .limit(N)
    )

    def jj(x):
        if what.model.__name__ == 'Cit':
            return x.cit.what
        if what.model.__name__ == 'Doc':
            return x.what

    return [
        ( jj(d), d.count)
        for d in q
    ]








Cit1 = Cit.alias()
Cit2 = Cit.alias()

def bib_network(filt, cited_ent='full_ref', min_cocit=5, limit=None):
    import networkx as nx
    from peewee import Expression
    
    # join it all together
    query = (Cit1
        .select(getattr(Cit1, cited_ent), getattr(Cit2, cited_ent), fn.COUNT().alias("count"))
        .join(Cit2, JOIN.INNER, on=(
            (Cit1.doc == Cit2.doc) & 
            (Cit1.full_ref != Cit2.full_ref) # not the same citations
        ))
        .join(Doc, on=(Cit1.doc == Doc.id))
        .group_by(getattr(Cit1, cited_ent), getattr(Cit2, cited_ent))
    )

    query = filt_it( query, filt )

    if min_cocit is not None:
        query = query.having(SQL('count') >= min_cocit)

    if limit is not None:
        query = query.limit(limit)

    g = nx.Graph()
    for r in query:
        g.add_edge(
            getattr(r, cited_ent),
            getattr(r.cit, cited_ent), 
            weight=r.count
        )
    return g