from .. import re
from ..exceptions import invalid_entry

class wos_doc:

    def __init__(self, raw):
        self.raw = raw

        self.journal = raw['SO'].lower()
        try:
            self.publish = int(raw['PY'])
        except ValueError:
            raise invalid_entry("doc year not valid")

        self.citing_authors = list(self.get_citing_authors())
        if not len(self.citing_authors):
            raise invalid_entry("No authors", self.raw['AU'].split(";"))

        self.uid = self.raw['UT']


    def generate_references(self):
        refs = self.raw['CR'].split(';')
        for r in refs:


            try:
                ref = wos_ref(r)

                if "*" in ref.author:
                    continue

                yield ref

            except invalid_entry:
                pass

    def get_citing_authors(self):
        authors = self.raw['AU'].split(";")
        for x in authors:
            x = x.strip().lower()
            x = re.sub("[^a-zA-Z\s,]+", "", x)  # only letters and spaces allowed

            xsp = x.split(", ")
            if len(xsp) != 2:
                yield x
                continue

            f, l = xsp[1], xsp[0]
            f = f[0]  # take only first initial of first name

            yield "%s, %s" % (l, f)


class wos_ref:

    def fixcitedauth(self):
        a = self.author.strip()
        if not len(a):
            raise invalid_entry("author too short")

        sp = a.lower().split()
        if len(sp) < 2:
            raise invalid_entry("author too short")
        if len(sp) >= 5:
            raise invalid_entry("author too long")

        l, f = a.lower().split()[:2]  # take first two words

        if len(l) == 1:  # sometimes last and first name are switched for some reason
            l, f = f, l

        f = f[0] + "."  # first initial

        a = ", ".join([l, f])  # first initial
        a = a.title()  # title format, so I don't have to worry later

        self.author = a

    def __init__(self, raw):




        yspl = re.split("((?:18|19|20)[0-9]{2})", raw)

        if len(yspl) < 2:
            raise invalid_entry("ref too short")

        self.author, self.publish = yspl[:2]
        self.fixcitedauth()

        self.publish = int(self.publish)

        self.full_ref = raw

        if 'DOI' not in self.full_ref and not (  # it's a book!
                len(re.findall(r', [Vv][0-9]+', self.full_ref)) or
                len(re.findall(r'[0-9]+, [Pp]?[0-9]+', self.full_ref))
        ):
            # full_ref = re.sub(r', [Pp][0-9]+', '', full_ref) # get rid of page numbers!

            self.full_ref = "|".join(  # splits off the author and year, and takes until the next comma
                [self.author] +
                [x.strip().lower() for x in self.full_ref.split(",")[2:3]]
            )
            self.type = 'book'

        else:  # it's an article!
            # just adds a fixed name and date to the front
            self.full_ref = "|".join(
                [self.author, str(self.publish)] +
                [",".join(x.strip() for x in self.full_ref.lower().split(",")[2:]).split(",doi")[0]]
            )
            self.type = 'article'