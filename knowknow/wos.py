
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



def import_from_txt( folder ):

