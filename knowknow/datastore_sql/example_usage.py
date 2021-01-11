# db.drop_tables([Doc,Cit,Keyword], safe=True)
# db.create_tables([Doc,Cit,Keyword], safe=True)

print(Doc.select().count(), 'docs already...')

txt_to_rows(
    "G:/My Drive/2020 ORGANISATION/1. PROJECTS/qualitative analysis of literature/110 CITATION ANALYSIS/000 data/sociology wos")

query = (Doc
         .select(Doc.id, fn.COUNT().alias("c"))
         .join(Cit)
         .switch(Doc)
         .group_by(Doc.id)
         )

j = 0
for i, x in enumerate(query):
    if j > 5:
        break

    j += 1

    print(x.id, x.c)

query = (Cit
         .select(fn.COUNT().alias("c"))
         .join(Doc, on=(Cit.doc == Doc.id))
         .switch(Cit)
         .group_by(Cit.full_ref, Doc.year)
         )

q2 = (Cit
      .select(Cit.full_ref, Doc.year, fn.COUNT().alias("c"))
      .join(Doc, on=(Cit.doc == Doc.id))
      .switch(Cit)
      .group_by(Cit.full_ref, Doc.year)
      .where(query >= 2)
      )

print(query.sql())
print(q2.sql())

query = (Cit
         .select(Cit, Doc, fn.COUNT().alias("c"))
         .join(Doc, on=(Cit.doc == Doc.id))
         .switch(Cit)
         .where(Cit.full_ref == 'Bourdieu, P.|distinction')
         .group_by(Doc.year)
         )

query = (Cit
         .select(Cit, Doc, fn.COUNT().alias("c"))
         .join(Doc, on=(Cit.doc == Doc.id))
         .switch(Cit)
         .group_by(Cit.full_ref, Doc.year)
         )

for i, x in enumerate(query):
    if i > 10:
        break
    print(x.full_ref, x.doc.year, x.c)

baseq = (Cit
         .select(Cit, Doc, fn.COUNT().alias("c"))
         .join(Doc, on=(Cit.doc == Doc.id))
         .switch(Cit))

journalq = (Doc
            .select(Doc.journal, fn.COUNT().alias("c"))
            .group_by(Doc.journal)
            )
print("\n".join(list("%s: %s" % (x.journal, x.c) for x in journalq)))
journals = [x.journal for x in journalq]


def get_jy(j, start=1950, end=2020):
    from collections import defaultdict
    jy = (Doc
          .select(Doc.year, fn.COUNT().alias("c"))
          .where(Doc.journal == j)
          .group_by(Doc.year)
          .order_by(Doc.year)
          )
    my_years = defaultdict(int, {
        d.year: d.c
        for d in jy
    })
    return [my_years[yy] for yy in range(start, end + 1)]


def docsByYear(filter=[], start=1950, end=2020):
    from collections import defaultdict
    jy = (Doc
          .select(Doc.year, fn.COUNT().alias("c"))
          .where(Doc.journal == j)
          .group_by(Doc.year)
          .order_by(Doc.year)
          )
    my_years = defaultdict(int, {
        d.year: d.c
        for d in jy
    })
    return [my_years[yy] for yy in range(start, end + 1)]


def get_ajy(a, j, start=1950, end=2020):
    from collections import defaultdict
    jy = (Doc
          .select(Doc.year, fn.COUNT().alias("c"))
          .where(Doc.journal == j)
          .group_by(Doc.year)
          .order_by(Doc.year)
          )
    my_years = defaultdict(int, {
        d.year: d.c
        for d in jy
    })
    return [my_years[yy] for yy in range(start, end + 1)]


from time import time

st = time()

for j in journals:
    (j, get_jy(j))

print("finished", time() - st)


def search_citing_authors(name):
    q = (Doc
         .select(Doc.first_author, fn.COUNT().alias("c"))
         .where(Doc.first_author.contains(name))
         .group_by(Doc.first_author)
         )

    return Counter({x.first_author: x.c for x in q})