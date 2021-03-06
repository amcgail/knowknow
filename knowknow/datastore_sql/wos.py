from dbmodel import Doc,Cit,Keyword,db
from peewee import *
from collections import Counter

from pathlib import Path
from hashlib import md5
import re

debug=False







def fixcitedauth(a):
    a = a.strip()
    if not len(a):
        return None

    sp = a.lower().split()
    if len(sp) < 2:
        return None
    if len(sp) >= 5:
        return None
    
    l, f = a.lower().split()[:2] # take first two words
    
    if len(l) == 1: # sometimes last and first name are switched for some reason
        l, f = f, l
        
    f = f[0] + "." # first initial
    
    a = ", ".join([l, f]) # first initial
    a = a.title() # title format, so I don't have to worry later
    
    if debug:
        print('cited author:', a)
        
    return a


def generate_references(refs,mydoc, **kwargs):
    cits = []
    for r in refs:
        yspl = re.split("((?:18|19|20)[0-9]{2})", r)

        if len(yspl) < 2:
            continue

        auth, year = yspl[:2]
        auth = auth.strip()
        
        if len(auth.split(" ")) > 5:
            continue
        
        year = int(year)

        if auth == "":
            continue

        auth = fixcitedauth( auth )
        if auth is None: # catching non-people, and not counting the citations
            continue
        
        full_ref = r
        
        if 'DOI' not in full_ref and not ( # it's a book!
            len(re.findall(r', [Vv][0-9]+', full_ref)) or
            len(re.findall(r'[0-9]+, [Pp]?[0-9]+', full_ref))
        ):
            #full_ref = re.sub(r', [Pp][0-9]+', '', full_ref) # get rid of page numbers!
            
            full_ref = "|".join( # splits off the author and year, and takes until the next comma
                [auth]+
                [x.strip().lower() for x in full_ref.split(",")[2:3]]
            )
        else: # it's an article!
            # just adds a fixed name and date to the front
            full_ref = "|".join(
                [auth, str(year)] +
                [",".join( x.strip() for x in full_ref.lower().split(",")[2:] ).split(",doi")[0]]
                )
                
        if debug:
            print('fix_refs_worked',auth,year,full_ref)
            
        c = Cit(
                doc = mydoc,
                author = auth,
                year = year,
                full_ref = full_ref
            )
        cits.append(c)
            
    Cit.bulk_create(cits, batch_size=100)
    for c in cits:
        yield c



def fixcitingauth(au):
    authors = au.split(";")
    for x in authors:
        x = x.strip().lower()
        x = re.sub("[^a-zA-Z\s,]+", "", x) # only letters and spaces allowed

        xsp = x.split(", ")
        if len(xsp) < 2:
            return xsp[0]
        elif len(xsp) > 2:
            print("Warning:", 'author with too many commas', x)
            continue
            raise Exception("author with too many commas", x)

        f, l = xsp[1], xsp[0]
        f = f[0] # take only first initial of first name

        yield "%s, %s" % (l,f)





def generate_documents(basedir):
    import sys
    import csv
    maxInt = sys.maxsize

    basedir = Path(basedir)

    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    dcount=0
    for i, path_to_data in enumerate( list(basedir.glob("**/*.txt")) ):
        if not( i < 50 ):
            continue
        print('source_file_%s: %s'%(i,path_to_data))

        from csv import DictReader


        with path_to_data.open('r', encoding='utf8', newline='') as in_f:
            docs = []
            kws = []
            for r in DictReader(in_f, delimiter='\t', quoting=3):
                auths = list(fixcitingauth(r['AU']))
                if not len(auths):
                    continue

                try:
                    year = int(r['PY'])
                except ValueError:
                    if r['EA'].strip() == '':
                        print("\n\n\n")
                        print(path_to_data)
                        print('ERR','year',r['PY'],'cnt',r['NR'],r['TI'])
                        continue
                        
                    year = int( r['EA'].split("/")[-1].split(" ")[-1] )

                doc = Doc.create(
                    title = r['TI'],
                    first_author = auths[0],
                    journal = r['SO'].lower().strip(),
                    year = year,
                    citcnt_external = int(r['NR'])
                )
                docs.append(doc)


                for kw in r['ID'].split(";"):
                    kw = kw.strip().lower()
                    if kw == '':
                        continue

                    kws.append(Keyword(
                        doc=doc,
                        text=kw
                    ))

                for ref in generate_references(r['CR'].split(';'), doc):
                    pass
                
                yield doc

            with db.atomic():
                Keyword.bulk_create(kws, batch_size=100)
           

def txt_to_rows( folder ):
    from pathlib import Path

    #wos_base = "path/to/wos/data"
    basedir = Path(folder)
    # ensures there are not overflow errors while importing large CSVs


    counter = 0
    for doc in generate_documents(folder):
        counter += 1

        if counter % 500 == 0:
            print('on doc %s' % counter)