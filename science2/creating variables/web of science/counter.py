from collections import Counter, defaultdict
from csv import DictReader
from random import shuffle

from lib import *
jw = load_variable("journals_jw")
wjournals = [w for j,w in jw]




cnt_ind = defaultdict(lambda:defaultdict(int))
track_doc = defaultdict(lambda:defaultdict(set))
cnt_doc = defaultdict(lambda:defaultdict(int))

def cnt(term, space, doc):
    # it's a set, yo
    track_doc[space][term].add(doc)
    # update cnt_doc
    cnt_doc[space][term] = len(track_doc[space][term])
    # update ind count
    cnt_ind[space][term] += 1
















# arghhh csv error
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


from pathlib import Path
basedir = Path("G:/My Drive/projects/qualitative analysis of literature/009 get everything from WOS")






def fix_auths(auths):
    for a in auths:
        a = a.strip()
        if not len(a):
            continue

        a = a.split()[0].lower()
        yield a


def fix_refs(refs):
    for r in refs:
        yspl = re.split("((?:18|19|20)[0-9]{2})", r)

        if len(yspl) < 2:
            continue

        auth, year = yspl[:2]
        auth = auth.strip()
        year = int(year)

        if auth == "":
            continue

        auth = "&".join( fix_auths( auth.split(",") ) )

        yield ( auth, year, r )







import re

dcount = 0
total_inserts = 0
to_inserts = []



book_groups = load_variable("groups.books")
article_groups = load_variable("groups.articles")
print(len(book_groups))
print(len(article_groups))

from collections import defaultdict

def get_reps(groups):
    ret = defaultdict(set)
    for k,v in groups.items():
        ret[v].add(k)
    ret = {
        k: list(v)[0]
        for k,v in ret.items()
    }
    return ret

book_group_reps = get_reps(book_groups)
article_group_reps = get_reps(article_groups)



multi_year = defaultdict(set)

for i, f in enumerate( list(basedir.glob("**/*.txt")) ):

    with f.open(encoding='utf8') as pfile:
        r = DictReader(pfile, delimiter="\t")
        rows = list(r)

    print("File %s/%s: %s" % (i, len(list(basedir.glob("**/*.txt"))),f))

    for i, r in enumerate(rows):

        if r['SO'] not in wjournals:
            continue

        if r['DT'] != "Article":
            continue

        refs = r["CR"].strip().split(";")
        refs = list( fix_refs(refs) )

        if not len(refs):
            continue

        #print(refs)

        dcount += 1
        if dcount % 10000 == 0:
            print("Document %s" % dcount)


        authors = r['AU'].split(";")
        authors = [x.strip() for x in authors]

        if False:
            for i in range(10):
                print("-"*20)


        uid = r['UT']


        try:
            int(r['PY'])
        except ValueError:
            continue



        for (auth,year,full_ref) in refs:
            if debug:
                print(full_ref)
            if 'DOI' not in full_ref and not len(re.findall(r', V[0-9]+, P[0-9]+', full_ref)):
                # splits off the author and year, and takes until the next comma
                full_ref = "|".join(
                    [auth]+
                    [x.strip().lower() for x in full_ref.split(",")[2:3]]
                )
            else:
                # just adds a fixed name and date to the front
                full_ref = "|".join(
                    [auth, str(year)] +
                    [",".join( full_ref.strip().lower().split(",")[2:] ).split(", doi")[0]]
                )
            if full_ref in book_groups:
                # retrieves retrospectively-computed groups
                full_ref = book_group_reps[
                    book_groups[full_ref]
                ]
                multi_year[full_ref].add(year)
            elif full_ref in article_groups:
                # retrieves retrospectively-computed groups
                full_ref = article_group_reps[
                    article_groups[full_ref]
                ]
            else:
                # a small minority, the ones which are dropped in this process anyways
                continue

            if debug:
                print(full_ref)
                print("--------------------")


            ref = (auth,year)

            cnt(r['SO'], 'fj', uid)
            cnt(int(r['PY']), 'fy', uid)
            cnt(ref[1], 'ty', uid)
            cnt((int(r['PY']), year), 'fy.ty', uid)
            cnt((r['SO'], year), 'fj.ty', uid)

            ref_str = "|".join( str(x) for x in ref )
            ref_str = full_ref.strip()
            #print(ref_str)

            name_blacklist = [
                "*us", 'Press', 'Knopf', '(January', 'Co', 'London', 'Bros', 'Books', 'Wilson','[anonymous]'
            ]
            if ref[0] in name_blacklist:
                continue
            if "*" in ref[0]:
                continue

            journal_keep = ["ETHNIC AND RACIAL STUDIES", "LAW & SOCIETY REVIEW", "DISCOURSE & SOCIETY", "SOCIOLOGICAL INQUIRY", "CONTRIBUTIONS TO INDIAN SOCIOLOGY", "SOCIETY & NATURAL RESOURCES", "RATIONALITY AND SOCIETY", "DEVIANT BEHAVIOR", "ACTA SOCIOLOGICA", "SOCIOLOGY-THE JOURNAL OF THE BRITISH SOCIOLOGICAL ASSOCIATION", "WORK EMPLOYMENT AND SOCIETY", "SOCIOLOGICAL METHODS & RESEARCH", "SOCIOLOGICAL PERSPECTIVES", "JOURNAL OF MARRIAGE AND FAMILY", "WORK AND OCCUPATIONS", "JOURNAL OF CONTEMPORARY ETHNOGRAPHY", "THEORY AND SOCIETY", "POLITICS & SOCIETY", "SOCIOLOGICAL SPECTRUM", "RACE & CLASS", "ANTHROZOOS", "LEISURE SCIENCES", "COMPARATIVE STUDIES IN SOCIETY AND HISTORY", "SOCIAL SCIENCE QUARTERLY", "MEDIA CULTURE & SOCIETY", "SOCIOLOGY OF HEALTH & ILLNESS", "SOCIOLOGIA RURALIS", "SOCIOLOGICAL REVIEW", "TEACHING SOCIOLOGY", "BRITISH JOURNAL OF SOCIOLOGY", "JOURNAL OF THE HISTORY OF SEXUALITY", "SOCIOLOGY OF EDUCATION", "SOCIAL NETWORKS", "ARMED FORCES & SOCIETY", "YOUTH & SOCIETY", "POPULATION AND DEVELOPMENT REVIEW", "SOCIETY", "JOURNAL OF HISTORICAL SOCIOLOGY", "HUMAN ECOLOGY", "INTERNATIONAL SOCIOLOGY", "SOCIAL FORCES", "EUROPEAN SOCIOLOGICAL REVIEW", "JOURNAL OF HEALTH AND SOCIAL BEHAVIOR", "SOCIOLOGICAL THEORY", "SOCIAL INDICATORS RESEARCH", "POETICS", "HUMAN STUDIES", "SOCIOLOGICAL FORUM", "AMERICAN SOCIOLOGICAL REVIEW", "SOCIOLOGY OF SPORT JOURNAL", "SOCIOLOGY OF RELIGION", "JOURNAL OF LAW AND SOCIETY", "GENDER & SOCIETY", "BRITISH JOURNAL OF SOCIOLOGY OF EDUCATION", "LANGUAGE IN SOCIETY", "AMERICAN JOURNAL OF ECONOMICS AND SOCIOLOGY", "ANNALS OF TOURISM RESEARCH", "SOCIAL PROBLEMS", "INTERNATIONAL JOURNAL OF INTERCULTURAL RELATIONS", "SOCIAL SCIENCE RESEARCH", "SYMBOLIC INTERACTION", "JOURNAL OF LEISURE RESEARCH", "ECONOMY AND SOCIETY", "INTERNATIONAL JOURNAL OF COMPARATIVE SOCIOLOGY", "SOCIAL COMPASS", "SOCIOLOGICAL QUARTERLY", "JOURNAL OF MATHEMATICAL SOCIOLOGY", "AMERICAN JOURNAL OF SOCIOLOGY", "REVIEW OF RELIGIOUS RESEARCH", "RURAL SOCIOLOGY", "JOURNAL FOR THE SCIENTIFIC STUDY OF RELIGION", "ARCHIVES EUROPEENNES DE SOCIOLOGIE", "CANADIAN JOURNAL OF SOCIOLOGY-CAHIERS CANADIENS DE SOCIOLOGIE", "POLISH SOCIOLOGICAL REVIEW"]
            if r['SO'] not in journal_keep:
                continue

            cnt((r['SO'],int(r['PY'])), 'jy', uid)

            for a in authors:
                cnt(a, 'a', uid)
                cnt((a,int(r['PY'])), 'ay', uid)
                cnt((a,r['SO']), 'aj', uid)
                cnt((a,r['SO'], int(r['PY'])), 'ajy', uid)

                cnt((a,ref_str), 'ac', uid)

            cnt(ref_str, 'c', uid)
            cnt((ref_str, int(r['PY'])), 'cy', uid)
            cnt((ref_str, r['SO']), 'cj', uid)
            cnt((ref_str, r['SO'], int(r['PY'])), 'cjy', uid)


            if not debug:
                to_insert = {
                    'doc':uid,
                    'a':authors,
                    'fy':int(r['PY']),
                    'j': r['SO'],
                    'ref':ref,
                    'ref_full': full_ref,
                    'refy': int(year)
                }
                to_inserts+=[to_insert]

                if len(to_inserts) > 5000:
                    BIB_DB.insert_many(to_inserts)

                    total_inserts += len(to_inserts)
                    to_inserts = []

                    print("%s bib_db entries created" % total_inserts)

# retrieve and use the MOST COMMON pub date for each
pubyears = {
    k:min(s) for k,s in multi_year.items()
    #if len(s)>1
}
varname = "pubyears.wos.all.uid"
save_variable(varname, pubyears)
print("saved %s" % varname)


if len(to_inserts):
    BIB_DB.insert_many(to_inserts)

    total_inserts += len(to_inserts)
    to_inserts = []

    print("final insert: %s bib_db entries created" % total_inserts)

















allowed_refs = Counter(dict(cnt_ind['c'].items())).most_common(1000)
allowed_refs = set( x[0] for x in allowed_refs )

print("Allowed references for cocitation analysis: %s" % len(allowed_refs))
print("Examples: %s" % str(list(allowed_refs)[:3]))

# enumerating cocitation for works with at least 10 citations
dcount = 0
refcount = 0

for i, f in enumerate(list(basedir.glob("**/*.txt"))):

    with f.open(encoding='utf8') as pfile:
        r = DictReader(pfile, delimiter="\t")
        rows = list(r)

    print("File %s/%s: %s" % (i, len(list(basedir.glob("*.txt"))), f))

    for i, r in enumerate(rows):

        if r['DT'] != "Article":
            continue

        refs = r["CR"].strip().split(";")
        refs = list(fix_refs(refs))

        if not len(refs):
            continue

        uid = r['UT']

        try:
            int(r['PY'])
        except ValueError:
            continue

        for ref in refs:

            ref_str = "|".join( str(x) for x in ref )
            if ref_str not in allowed_refs:
                continue

            for ref2 in refs:
                ref_str2 = "|".join( str(x) for x in ref2 )

                if ref_str2 <= ref_str:
                    continue
                if ref_str2 not in allowed_refs:
                    continue

                cnt((ref_str,ref_str2), 'cc', uid)
                refcount += 1
                if refcount % 10000 == 0:
                    print("%s cocitations logged" % refcount)

















print("---------------------DOC COUNTER-----------------------------")
for ln in cnt_doc.keys():
    print("COUNTER %s" % ln)
    ln_c = Counter(dict(cnt_doc[ln].items()))
    for what, c in ln_c.most_common(5):
        print("(%s) %s" % (c, str(what)))
print("---------------------IND COUNTER-----------------------------")
for ln in cnt_ind.keys():
    print("COUNTER %s" % ln)
    ln_c = Counter(dict(cnt_ind[ln].items()))
    for what, c in ln_c.most_common(5):
        print("(%s) %s" % (c, str(what)))

import pickle
#cnt_doc = {k:{what:c for what,c in cnt_doc[k].items()} for k in cnt_doc}
#todelete = set(c for c in cnt_doc['c'] if cnt_doc['c'] > 1)
#del cnt_doc['c'][]
pickle.dump({k:{what:c for what,c in cnt_doc[k].items()} for k in cnt_doc}, open("cnt_doc_fullUnique.pickle", 'wb'))
pickle.dump({k:{what:c for what,c in cnt_ind[k].items()} for k in cnt_ind}, open("cnt_ind_fullUnique.pickle", 'wb'))