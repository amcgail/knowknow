import sqlalchemy

def fix_refs(refs):
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

        if use_included_citations_filter:
            if full_ref not in included_citations:
                continue
            
        # implement grouping of references
        if groups is not None:
            if full_ref in groups:
                # retrieves retrospectively-computed groups
                full_ref = group_reps[
                    groups[full_ref]
                ]
            elif full_ref in ysumc and '[no title captured]' not in full_ref:
                # a small minority, the ones which are dropped in this process anyways
                #print('not in grouping')
                raise Exception(full_ref, "not in grouping!")
            else:
                continue
                
        if debug:
            print('fix_refs_worked',auth,year,full_ref)
        yield ( auth, year, full_ref )
        

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

        articles = df
        articles.to_sql('articles', index=False, con=engine, if_exists=behaviour)

        authors = []
        citations = []
        articles = []
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

            citing_authors = list(fixcitingauth())
            
            curr_authors = pandas.DataFrame({'name': names, 'fullname': fullnames, 'idx': range(len(names))})
            curr_authors['name_key'] = list(fixcitingauth())
            
            authors.append(curr_authors)

            curr_cit = fix_refs( row['CR'].split(';') )
            citations.append(pandas.DataFrame({
                'auth':[x[0] for x in curr_cit],
                'year':[x[1] for x in curr_cit],
                'full_ref':[x[2] for x in curr_cit]
            }))
            citations[-1]['citing_work'] = uid

        authors = pandas.concat(authors)
        citations = pandas.concat(citations)

        authors.to_sql('authors', index=False, con=engine, if_exists=behaviour)
        citations.to_sql('citations', index=False, con=engine, if_exists=behaviour)
        