import string_grouper
import editdistance
from .dbquery import *


import pandas as pd

#class Groups:

# the final variable we are constructing
groups = {}

# tracks the last group-id assigned
new_gid = 0


# what are all the cited references??
dd = count_docs(
    by={"ref":Cit.full_ref},
    min_count = 2
)

strings = list(dd['ref'])

len(strings)

def isarticle(x):
    sp = x.split("|")
    if len(sp) < 2:
        return False
    
    try:
        int(sp[1])
        return True
    except ValueError:
        return False

strings = [x for x in strings if '[no title captured]' not in x]
articles = [x for x in strings if isarticle(x)]
books = [x for x in strings if not isarticle(x)]

print("%s articles, %s books to group" % (len(articles), len(books)))

# grouping books

# this cell may take quite a while to run.
# on Intel i7-9700F this runs in about a minute on 185k names.

books_grouped = string_grouper.match_strings(
    pd.Series(books), 
    number_of_processes=8, 
    min_similarity=0.7
)

from collections import defaultdict

books_grouped[(books_grouped.similarity<1-1e-8)].sort_values("similarity")

# for books, we require that the authors are no more than 1 edit from each other
# even after limiting the comparisons necessary, this takes about 20s on Intel i7-9700F

ft = defaultdict(set)

for i,r in books_grouped.iterrows():
    ls = r.left_side
    rs = r.right_side
    
    if ls == rs:
        continue
    
    la = ls.split("|")[0]
    ra = rs.split("|")[0]
    
    if editdistance.eval(la,ra) > 1:
        continue
    
    ft[ls].add(rs)
    ft[rs].add(ls)
    
print("%s books have some connection to others in a group" % len(ft))

# assigns group-ids based on the relational structure derived thus far
# the code propagates ids through the network, assuming transitivity of equality

def traverse(x, gid):
    global groups
    groups[x] = gid
    
    neighbors = ft[x]
    for n in neighbors:
        if n not in groups:
            traverse(n, gid)
      
for i,k in enumerate(books):
    if k in groups:
        continue
        
    traverse(k, new_gid)
    new_gid += 1

len(set(groups.values()))

Counter(gid for x,gid in groups.items() if len(x.split("|"))==2).most_common(10)

# grouping articles

# this cell may take quite a while to run.
# on Intel i7-9700F this runs in five minutes on 234k entries.

articles_grouped = string_grouper.match_strings(
    pd.Series(articles), 
    number_of_processes=8, # decrease this number to 1 or 2 for slower computers or laptops (the fan might start screaming)
    min_similarity=0.8 # the similarity cutoff is tighter for articles than for books
)

articles_grouped[(articles_grouped.similarity<1-1e-8)].sort_values("similarity")

# for articles, we require that the entire citations is only 1 edit apart.
# even after limiting the comparisons necessary, this takes about 20s on Intel i7-9700F

# this cell produces the `ft` variable, which maps from each term to the set of terms equivalent. I.e., `ft[A] = {B1,B2,B3}`

ft = defaultdict(set)

for i,r in articles_grouped.iterrows():
    ls = r.left_side
    rs = r.right_side
    
    if ls == rs:
        continue
    
    la = ls.split("|")[0]
    ra = rs.split("|")[0]
        
    if editdistance.eval(ls,rs) > 2:
        continue
    
    ft[ls].add(rs)
    ft[rs].add(ls)
    #print(ls,"|||",rs)

print("%s articles have some connection to others in a group" % len(ft))

# assigns group-ids based on the relational structure derived thus far
# the code propagates ids through the network, assuming transitivity of equality

def traverse(x, gid):
    global groups
    groups[x] = gid
    
    neighbors = ft[x]
    for n in neighbors:
        if n not in groups:
            traverse(n, gid)

for i,k in enumerate(articles):
    if k in groups:
        continue
        
    traverse(k, new_gid)
    new_gid += 1

# this line will break execution if there aren't as many groups assigned as we have articles and books
assert( len(articles) + len(books) == len(groups) )

len(set(groups.values()))

len(set(groups.values())) - len(articles)

len(set(groups.values())) - len(books) - len(articles)

len(books)

len(articles)

from collections import defaultdict

# for quicker access
counts = {
    x.ref: int(x['count'])
    for i,x in dd.iterrows()
}

group_members = defaultdict(set)
for k,v in groups.items():
    group_members[v].add(k)

def get_reps(groups):

    for k,v in group_members.items():
        ret[k] = max(v, key=lambda x: counts[x])

    return ret

group_reps = get_reps(groups)

updated = 0
for i, cit in enumerate(Cit.select()):
    if cit.full_ref not in groups:
        continue

    grp = groups[cit.full_ref]

    if len(group_members[grp]) < 2:
        continue

    new_name = group_reps[grp]

    if new_name == cit.full_ref:
        continue

    #print('updating %s with %s' % (cit.full_ref, new_name))
    #break
    cit.full_ref = new_name
    cit.author = new_name.split("|")[0]
    cit.save()
    updated += 1

    if updated % 1000 == 0:
        print("Updated %s citations..."%updated)

database_name = 'wos'

# saving the variable for later
save_variable("%s.groups" % database_name, groups)
save_variable("%s.group_reps" % database_name, groups)