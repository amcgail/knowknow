

def fixcitingauth():
    authors = r['AU'].split(";")
    for x in authors:
        x = x.strip().lower()
        x = re.sub("[^a-zA-Z\s,]+", "", x) # only letters and spaces allowed

        xsp = x.split(", ")
        if len(xsp) < 2:
            return xsp[0]
        elif len(xsp) > 2:
            raise Exception("author with too many commas", x)

        f, l = xsp[1], xsp[0]
        f = f[0] # take only first initial of first name

        yield "%s, %s" % (l,f)


def _txt_to_rows( folder ):
    from pathlib import Path

    #wos_base = "path/to/wos/data"
    wos_base = "G:/My Drive/2020 ORGANISATION/1. PROJECTS/2. CURRENT/qualitative analysis of literature/pre 5-12-2020/009 get everything from WOS"
    basedir = Path(wos_base)
    # ensures there are not overflow errors while importing large CSVs

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

    db_path = 'wos.db'
    engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')

    dcount=0
    for i, path_to_data in enumerate( list(basedir.glob("**/*.txt")) ):
        print('source_file_%s: %s'%(i,path_to_data))

        #path_to_data = '1-500.txt'
        if i == 0:
            behaviour = 'replace'  # fail, append
        else:
            behaviour = 'append'

        from csv import DictReader
        with path_to_data.open('r', encoding='utf8', newline='') as in_f:
            df = pandas.DataFrame(list(DictReader(in_f, delimiter='\t')))
        # df.to_sql('wos', engine)

        articles = pandas.DataFrame()
        articles['title'] = df['TI']

        articles['citation_count'] = df['NR']
        articles['id'] = df['OI']
        articles['year'] = df['PY']
        articles['journal'] = df['PU']
        articles['myuid'] = articles.title.apply(lambda x: md5(x.encode()).hexdigest())

        articles.to_sql('articles', index=False, con=engine, if_exists=behaviour)

        authors = []
        citations = []
        for idx, row in df.iterrows():
            if pandas.isna(row.AU) or pandas.isna(row.CR):
                continue
            uid = articles['myuid'].loc[idx]
            names = row['AU'].split(';')
            fullnames = row['AF'].split(';')

            names = [x.strip() for x in names]
            fullnames = [x.strip() for x in fullnames]

            if len(names) != len(fullnames):
                print("Skipping bc. name discrepancy")
                continue

            curr_authors = pandas.DataFrame({'name': names, 'fullname': fullnames, 'idx': range(len(names))})
            curr_authors['work_id'] = uid
            authors.append(curr_authors)

            curr_cit = row['CR'].split(';')
            citations.append(pandas.DataFrame({'cited_work': curr_cit}))
            citations[-1]['citing_work'] = uid

        authors = pandas.concat(authors)
        citations = pandas.concat(citations)

        authors.to_sql('authors', index=False, con=engine, if_exists=behaviour)
        citations.to_sql('citations', index=False, con=engine, if_exists=behaviour)



        # processes WoS txt output files one by one, counting relevant cooccurrences as it goes


dcount=0
for i, f in enumerate( list(basedir.glob("**/*.txt")) ):

    with f.open(encoding='utf8') as pfile:
        df = pandas.DataFrame(list(DictReader(in_f, delimiter='\t')))
        
    if i % 50 == 0:
        print("File %s/%s: %s" % (i, len(list(basedir.glob("**/*.txt"))),f.name))
        
    articles = []
    citations = []
    documents = []
    
    for idx, row in df.iterrows():
        if pandas.isna(row.AU) or pandas.isna(row.CR):
            continue
        uid = articles['myuid'].loc[idx]
        names = row['AU'].split(';')
        fullnames = row['AF'].split(';')
        
        if row['DT'] != "Article":
            continue

        names = [x.strip() for x in names]
        fullnames = [x.strip() for x in fullnames]

        if len(names) != len(fullnames):
            print("Skipping bc. name discrepancy")
            continue
            
        dcount += 1
        if dcount % 10000 == 0:
            print("Document: %s" % dcount)
            print("Citations: %s" % len(cnt_doc['c']))
            
        if debug:
            print("DOCUMENT %s" % dcount)
            if dcount > 10:
                raise

        refs = row["CR"].strip().split(";")
        refs = list( fix_refs(refs) )      

        if not len(refs):
            continue          
                
        curr_authors = pandas.DataFrame({'name': names, 'fullname': fullnames, 'idx': range(len(names))})
        curr_authors['work_id'] = uid
        authors.append(curr_authors)
        
        curr_cit = row['CR'].split(';')
        articles.append(pandas.DataFrame({
            
        }))
        citations.append(pandas.DataFrame({'cited_work': curr_cit}))
        citations[-1]['citing_work'] = uid
       
            
        citing_authors = list(fixcitingauth())
        if not len(citing_authors):
            continue
        
        if debug:
            print("citing authors: ", citing_authors)

        if False:
            for i in range(10):
                print("-"*20)


        uid = r['UT']


        try:
            int(r['PY'])
        except ValueError:
            continue

        r['SO'] = r['SO'].lower() # REMEMBER THIS! lol everything is in lowercase... not case sensitive
        
        if use_included_journals_filter and r['SO'].lower() not in journal_keep:
            continue     
            
    authors = pandas.concat(authors)
    citations = pandas.concat(citations)
    

                    
            ref = (auth,year)
           
        
            if ref[0] in name_blacklist:
                continue
            if "*" in ref[0]:
                continue
                
            # BEGIN COUNTING!!!
            
            multi_year[full_ref][year] += 1
                
            cnt(r['SO'], 'fj', uid)
            cnt(int(r['PY']), 'fy', uid)
            cnt(ref[1], 'ty', uid)
            cnt((full_ref, int(r['PY'])), 'c.fy', uid)
            cnt((full_ref, r['SO']), 'c.fj', uid)

            cnt((r['SO'],int(r['PY'])), 'fj.fy', uid)

            cnt(full_ref, 'c', uid)
            
            if not RUN_EVERYTHING:
                continue
                
            cnt((int(r['PY']), year), 'fy.ty', uid)
            cnt((r['SO'], year), 'fj.ty', uid)
                    
            cnt(auth, 'ta', uid)
            cnt((int(r['PY']),auth), 'fy.ta', uid)
            cnt((r['SO'],auth), 'fj.ta', uid)
            
            # first author!
            ffa = citing_authors[0]
            cnt(ffa, 'ffa', uid)
            cnt((ffa,int(r['PY'])), 'ffa.fy', uid)
            cnt((ffa,r['SO']), 'ffa.fj', uid)
            cnt((full_ref,ffa), 'c.ffa', uid)
            #cnt((ffa,r['SO'], int(r['PY'])), 'ffa.fj.fy', uid)

            for a in citing_authors:
                cnt(a, 'fa', uid)
                cnt((a,int(r['PY'])), 'fa.fy', uid)
                cnt((a,r['SO']), 'fa.fj', uid)
                #cnt((a,r['SO'], int(r['PY'])), 'fa.fj.fy', uid)

                cnt((full_ref,a), 'c.fa', uid)
            
            cnt((full_ref, int(r['PY']), r['SO']), 'c.fy.j', uid)

        
        
        
        
        
def import_from_txt( folder ):

