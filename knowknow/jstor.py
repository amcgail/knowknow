__all__ = [
    "file_iterator", "ParseError", "debug"
]



from nltk import sent_tokenize


# XML parser
from lxml.etree import _ElementTree as ElementTree
from lxml import etree
recovering_parser = etree.XMLParser(recover=True)

debug = False


from zipfile import ZipFile



# A WHOLE SHIT-TON OF HELPERS


import sys; sys.path.append(__file__.split("knowknow")[0])
from knowknow import *

import pickle
import calendar


def getOuterParens(x):
    paren_level = 0
    buffer = ""
    rets = []
    i_start = None

    for i, c in enumerate(x):
        if c == "(":
            if paren_level == 0:
                i_start = i
            paren_level += 1
        if c == ")":
            paren_level -= 1

            if paren_level == 0:
                rets.append( ( i_start, buffer[1:] ) )
                buffer = ""
        if paren_level > 0:
            buffer += c

    return rets

# note that 6 is a common mistake for vowels!?
name = r"[A-Z][A-Za-z6'`-̈]+"
# we allow only a single break in the name
name = name + r"\s?" + r"[A-Za-z6'`-̈]*"
fname = r"(?:{name}(?:\s{name})?|{name} et al.?)".format(**locals())
names = r"(?P<names>{fname}(?:, {fname})*(?:,? (and|&) {fname})?)".format(**locals())
year = r"(?P<year>(((\[[0-9]{4}\]\s?)?[0-9]{4})[abcdef]?(, )?)+)"
ppnum = r"pp?\.? (?P<pnum>[0-9\-–—]+)"
pnum = r"(?P<pnum>[0-9\-–—]+)"
prefix = r"(?:(?P<prefix>((see|see also|listed in|and|e.g.|cf.|c.f.),?\s?)+) )?"
postfix = r"(?:(?P<postfix>,?.*))?"

regex_in_order = [
    r"^{prefix}{names} {year}, {ppnum}{postfix}$".format(**locals()),
    r"^{prefix}{names} {year}{postfix}$".format(**locals()),
    r"^{prefix}{names}, {year}: {pnum}{postfix}$".format(**locals()),
    r"^{prefix}{names}, {year}{postfix}$".format(**locals()),
    r"^{prefix}{names} \[{year}\]{postfix}$".format(**locals()),
    r"^{prefix}{names} \[{year}, [^\]]\]{postfix}$".format(**locals()),
]

#print(regex_in_order)

name_blacklist = calendar.month_name[1:]






faulty_years = set()

def citation_iterator(t):
    citations = extractCitation(t)
    for c in citations:
        name = c['names']
        name = name.replace("'s","") # change "Simmel's" to "Simmel"
        
        years = c['year'].split(",")
        for y in years:
            y = y.strip() # get rid of the extra spacing
            y = re.findall(r'[0-9]{4}',y) # this strips out the 'a' in '1995a'
            if not len(y):
                faulty_years.add(c['year'])
                continue
            
            y = y[0]
            yield "%s (%s)" % (name, y)


def extractCitation(t):
    t = re.sub(r"[-]\n", "", t)
    t = re.sub(r"\s+", " ", t)

    # try splitting it both ways,
    # just see which gets more hits

    tparts = t.split(";")
    #print("A")
    ret = []
    for tpart in tparts:
        tpart = tpart.strip()
        #print("Aa")
        # I guess this causes infinite looping??
        if len(tpart.split(",")) >= 4:
            continue
        #print("Ab")
        #print("Checking", tpart)
        for rgxi, rgx in enumerate(regex_in_order):
            mtch = re.match(rgx, tpart)
            if mtch:
                d = mtch.groupdict()
                if d['names'] in name_blacklist:
                    continue
                ret.append( d )
    #print("B")
    tparts2 = t.split(",")

    ret2 = []
    for tpart in tparts2:
        tpart = tpart.strip()

        #print("Checking", tpart)
        for rgx in regex_in_order:
            mtch = re.match(rgx, tpart)
            if mtch:
                d = mtch.groupdict()
                if d['names'] in name_blacklist:
                    continue
                ret2.append( d )
    #print("C")
    if len(ret2) > len(ret):
        return ret2
    return ret

def extractCitations(strs):
    results = []
    for s in strs:
        result = extractCitation(s)
        if result is not None:
            results.append(result)
    return results











def basic_ocr_cleaning(x):
    # remove multiple spaces in a row
    x = re.sub(r" +", ' ', str(x))
    # remove hyphenations [NOTE this should be updated, with respect to header and footer across pages...]
    x = re.sub(r"([A-Za-z]+)-\s+([A-Za-z]+)", "\g<1>\g<2>", x)
    
    x = x.strip()
    return x

def get_content_string(ocr_string):
    docXml = etree.fromstring(ocr_string, parser=recovering_parser)
    pages = docXml.findall(".//page")

    page_strings = []
    for p in pages:
        if p.text is None:
            continue
        page_strings.append(p.text)

    secs = docXml.findall(".//sec")

    for s in secs:
        if s.text is None:
            continue
        if s.text.strip() == '':
            try_another = etree.tostring(s, encoding='utf8', method='text').decode("utf8").strip()
            #print(try_another)
            if try_another == '':
                continue

            page_strings.append(try_another)
        else:
            page_strings.append(s.text.strip())

    return basic_ocr_cleaning( "\n\n".join(page_strings) )


































romanChars = "LIXMCV"

recovering_parser = etree.XMLParser(recover=True)

if False:
    import spacy

    nlp = spacy.load('en')
    import neuralcoref

    neuralcoref.add_to_pipe(nlp)

def untokenize(tokens):
    import string
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def all_files( dirs ):
    
    ocr_dirs = [
        Path(d).join("ocr")
        for d in dirs
    ]

    return chain.from_iterable([
        glob(join(d, "*.txt"))
        for d in ocr_dirs
    ])

def isRoman(x):
    if not type(x) == str:
        return False

    return all( y in romanChars for y in x.upper() )

def num_pdf2page(pdf_num, page_map):
    import roman
    beforeMe = [ x for x in page_map.items() if x[0] <= pdf_num ]
    if not len(beforeMe):
        return -1
    lastBeforeMe = max( beforeMe, key=lambda x: x[0] )

    afterMe = [x for x in page_map.items() if x[0] >= pdf_num]
    if not len(afterMe):
        return -1
    firstAfterMe = min(afterMe, key=lambda x: x[0])

    if isRoman(lastBeforeMe[1]) and isRoman(firstAfterMe[1]):
        return roman.toRoman( pdf_num - lastBeforeMe[0] + roman.fromRoman(lastBeforeMe[1].upper()) )

    elif not isRoman(lastBeforeMe[1]) and not isRoman(firstAfterMe[1]):
        return pdf_num - lastBeforeMe[0] + int(lastBeforeMe[1])

    else:
        return -1




def follow_expand( from_what, follow_what ):
    found = [x for x in from_what.children if x.dep_ == follow_what]
    return [
        " ".join(str(x) for x in one.subtree)
        for one in found
    ]

def fix_paragraph(x):
    # replace what look like hyphenations
    x = re.sub(r"(\S+)-[\n\r]\s*(\S+)", r"\1\2", x)

    # replace newlines with spaces
    x = re.sub(r"[\n\r]", " ", x)

    # replace multiple spaces
    x = re.sub(r" +", " ", x)

    return x

class Sentence:
    def __init__(self, content=None ,lines=None, words=None):
        self.lines = lines
        self.content = content
        self.words = words
        self.type = None

        if self.words is not None:
            self.content = untokenize(self.words)

        # process hyphenation
        self.content = fix_paragraph(self.content)

    def parse(self):

        self.doc = nlp(self.content)
        self.root_verb = [word for word in self.doc if word.head == word ][0]

        ".+ is one thing and .+ is quite another"
        ".+ is one thing and .+ is another"
        ".+ is the same as .+"

        if str(self.root_verb) == "is":

            nsubj = follow_expand( self.root_verb, 'nsubj' )
            attr = follow_expand( self.root_verb, 'attr' )
            advc = follow_expand( self.root_verb, 'advcl' )

            if len(nsubj) == 1 and len(attr) == 1:
                self.type = "identification"
                self.args = [
                    nsubj[0],
                    attr[0],
                    advc
                ]
                print("X(%s) IS Y(%s)" % tuple(self.args[:2]))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.content

class Line:
    def __init__(self, content=None, page=None):
        self.content = content.strip()
        self.page = None
        self.type = None

        #if len(self.content) < 5:
        #    print(self.content)

        if self.content == "":
            self.type = "empty"

        try:
            numerical_content = int(self.content)
            self.type = "page_number"
            self.value = numerical_content
        except ValueError:
            pass

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return self.content

class Page:

    @classmethod
    def from_lines(cls, lines):
        p = Page()

        for line in lines:
            p.lines.append(Line(
                # content
                content=line,
                # and context
                page=p
            ))

        p.extract_sentences()

        return p

    @classmethod
    def from_text(cls, text):
        return Page.from_lines(re.split("[\n\r]+", text))

    def __init__(self):
        self.lines = []
        self.start_stub = None
        self.end_stub = None
        self.full_sentences = None


    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "\n".join( map(str,self.lines) )

    def extract_sentences(self):
        from nltk import sent_tokenize, word_tokenize

        sents = [ word_tokenize(ss) for ss in sent_tokenize( str(self) ) ]

        if not len(sents):
            raise ParseError("Sentences not found")

        full_sentences = [ Sentence(words=sent_words) for sent_words in sents[1:-1] ]
        start_stub = sents[0]
        end_stub = sents[-1]

        self.start_stub = start_stub
        self.end_stub = end_stub
        self.full_sentences = full_sentences

    @property
    def content(self):
        return str(self)

class ParseError(Exception):
    pass

ALL_UNIVERSITIES = {
    x[1].upper().strip() 
    for x in creader(
        Path(BASEDIR, "external-data", "world-universities.csv")
            .open('r', encoding='UTF8')
    )
}
max_uni_length = max( len(x) for x in ALL_UNIVERSITIES )







def clean_metadata( doi, metadata_str ):

    try:
        metaXml = etree.fromstring(metadata_str, parser=recovering_parser)
    except OSError:
        raise ParseError("NO METADATA!", fn)

    def findAndCombine(query):
        return ";".join([" ".join(x.itertext()) for x in metaXml.findall(query)]).strip()

    metadata = {}
    metadata['type'] = metaXml.get("article-type")

    metadata['doi'] = doi
    # print(doi)

    metadata['title'] = findAndCombine(".//article-title")
    metadata['journal'] = findAndCombine(".//journal-title")
    metadata['publisher'] = findAndCombine(".//publisher-name")

    metadata['abstract'] = findAndCombine(".//article-meta//abstract")

    auth = []
    for group in metaXml.findall(".//contrib"):
        myname = " ".join(x.strip() for x in group.itertext())
        myname = " ".join(re.split("\s+", myname)).strip()
        auth.append(myname)
    metadata['authors'] = auth

    if len(auth) == 0 and False:
        print(doi)
        print(auth)
        print(metadata['title'])

    metadata['month'] = findAndCombine(".//article-meta//month")

    metadata['year'] = findAndCombine(".//article-meta//year")
    if ";" in metadata['year']:
        metadata['year'] = metadata['year'].split(";")[0]
    try:
        metadata['year'] = int(metadata['year'])
    except:
        raise ParseError("No valid year found")

    metadata['volume'] = findAndCombine(".//article-meta//volume")
    metadata['issue'] = findAndCombine(".//article-meta//issue")

    metadata['fpage'] = findAndCombine(".//article-meta//fpage")
    metadata['lpage'] = findAndCombine(".//article-meta//lpage")

    return metadata
















class Document:
    def __init__(self, metadata):
        self.pages = []
        self.metadata = metadata

        self.type = self.metadata['type']

        lt = self['title'].lower()
        if lt in ["book review", 'review essay']:
            self.type = 'book-review'
        if "Commentary and Debate".lower() == lt or 'Comment on'.lower() in lt:
            self.type = 'commentary'
        if lt.find('reply to') == 0:
            self.type = 'commentary'
        if 'Erratum'.lower() in lt:
            self.type = 'erratum'

    @classmethod
    def from_db(cls, _id):
        raise Exception("Haven't done yet!")

    @classmethod
    def from_file(cls, fn, complex_parsing=True):
        metadata = cls.get_metadata(fn)
        page_strings = cls.get_page_strings(fn)

        page_strings = list(map(basic_ocr_cleaning, page_strings))
        page_strings = [x for x in page_strings if x != ""]
        if not len(page_strings):
            raise ParseError("Empty document")

        return Document.from_pages(page_strings, metadata=metadata, complex_parsing=complex_parsing)

    @classmethod
    def from_pages(cls, page_strings, metadata={}, complex_parsing=True):
        d = Document( metadata )

        if not len(page_strings):
            raise ParseError("No pages...")

        d.pages = [Page.from_lines(re.split(r"[\n\r]+", page)) for page in page_strings]

        if complex_parsing:
            if d.type == 'research-article':
                try:
                    d.find_bibliography()
                    d.extract_headers_footers()
                    d.parse_bibliography()
                except ParseError:
                    print("Couldn't extract bib and headers from ", d['doi'])
                    raise

        return d

    @property
    def sentences(self):
        for i, p in enumerate(self.pages):
            if i > 0:
                yield Sentence( words = self.pages[i-1].end_stub + self.pages[i].start_stub )

            p.extract_sentences()
            for s in p.full_sentences:
                yield s

    @classmethod
    def get_page_strings(cls, fn):
        docXml = etree.parse(fn, parser=recovering_parser)
        pages = docXml.findall(".//page")

        page_strings = []
        for p in pages:
            if p.text is None:
                continue
            page_strings.append(p.text)

        secs = docXml.findall(".//sec")

        for s in secs:
            if s.text is None:
                continue
            if s.text.strip() == '':
                try_another = etree.tostring(s, encoding='utf8', method='text').decode("utf8").strip()
                #print(try_another)
                if try_another == '':
                    continue

                page_strings.append(try_another)
            else:
                page_strings.append(s.text.strip())

        return page_strings

    @classmethod
    def get_metadata(cls, fn):

        my_name = ".".join(fn.split(".")[:-1])
        doi = my_name.split("-")[-1].replace("_", "/")

        metadataFn = "%s.xml" % my_name
        metadataFn = join(dirname(metadataFn), "..", "metadata", basename(metadataFn))


        try:
            metaXml = etree.parse(metadataFn, parser=recovering_parser)
        except OSError:
            raise ParseError("NO METADATA!", fn)
            
        assert (isinstance(metaXml, ElementTree))

        def findAndCombine(query):
            return ";".join([" ".join(x.itertext()) for x in metaXml.findall(query)]).strip()

        metadata = {}
        metadata['type'] = metaXml.getroot().get("article-type")

        metadata['doi'] = doi
        # print(doi)

        metadata['title'] = findAndCombine(".//article-title")
        metadata['journal'] = findAndCombine(".//journal-title")
        metadata['publisher'] = findAndCombine(".//publisher-name")

        metadata['abstract'] = findAndCombine(".//article-meta//abstract")

        auth = []
        for group in metaXml.findall(".//contrib"):
            myname = " ".join(x.strip() for x in group.itertext())
            myname = " ".join(re.split("\s+", myname)).strip()
            auth.append(myname)
        metadata['authors'] = auth

        if len(auth) == 0:
            print(doi)
            print(auth)
            print(metadata['title'])

        metadata['month'] = findAndCombine(".//article-meta//month")

        metadata['year'] = findAndCombine(".//article-meta//year")
        if ";" in metadata['year']:
            metadata['year'] = metadata['year'].split(";")[0]
        try:
            metadata['year'] = int(metadata['year'])
        except:
            raise ParseError("No valid year found")

        metadata['volume'] = findAndCombine(".//article-meta//volume")
        metadata['issue'] = findAndCombine(".//article-meta//issue")

        metadata['fpage'] = findAndCombine(".//article-meta//fpage")
        metadata['lpage'] = findAndCombine(".//article-meta//lpage")

        return metadata

    @property
    def num_pages(self):
        return len(self.pages)

    def apply_page_map(self, page_map):
        self.page_map = page_map

        for i, p in enumerate(self.pages):
            p.page_mapped = num_pdf2page( pdf_num=i, page_map=self.page_map)

    def extract_headers_footers(self):
        if self.type != 'research-article':
            print("Unimplemented: extracting headers and footers on non-research-article")
            return

        if not len(self.pages):
            raise ParseError()

        # extracting title, heading stuff on first page, and abstract
        fp = str(self.pages[0]).upper()
        found = []

        f = fp.find( self.metadata['title'].upper() )
        found.append( (f, len(self.metadata['title'])) )

        for a in self.metadata['authors']:
            f = fp.find(a.upper())
            found.append( (f, len(a)) )


        # to get the university, not included in most metadatas...
        found_uni = []
        for u in ALL_UNIVERSITIES:
            found.append( ( fp.find(u), len(u) ) )

        # limit to those strings which were actually found
        found = list(filter(lambda x: x[0]>=0, found))

        if len(found):
            # these strings should be "end to end", i.e. less than 5 characters separating them
            found_ends = [ x[0]+x[1] for x in found ]
            found_min_beginning = min( x[0] for x in found )
            found = list(filter(lambda x: any( abs(x[0]-y) < 5 for y in found_ends ) or x[0] == found_min_beginning, found))

            # we then cut after these metadata fields!
            doc_start = max(found, key=lambda x: x[0]+x[1])
            doc_start = doc_start[0] + doc_start[1]

            newfp_text = str(self.pages[0])[doc_start:].strip()
            #newfp = Page.from_text(newfp_text)
            self.pages[0] = newfp_text




        # extracting headers via voting
        num_identical_necessary = len(self.pages)/4

        flines = [ str(x).split()[:20] for x in self.pages ]
        grams = Counter(
            tuple(x[:i])
            for i in range(1,15)
            for x in flines
        )

        candidates = set(x for x,c in grams.items() if c > num_identical_necessary)
        maximal_candidates = []
        for c in candidates:
            keep=True
            for c2 in candidates:
                if len(c) >= len(c2):
                    continue
                if c[:min(len(c),len(c2))] == c2[:min(len(c),len(c2))]:
                    keep=False

            if keep:
                maximal_candidates.append(c)

        self.headers = [" ".join(x) for x in maximal_candidates]
        for h in self.headers:

            # don't want to get rid of the title on the first page.
            # that's the job for another algorithm
            # therefore, the [1:]
            for i,p in enumerate(self.pages[1:]):
                try:
                    if str(p).index(h) < 30:
                        self.pages[i] = Page.from_text( str(p).replace(h, "").strip() )
                except ValueError:
                    continue

    def parse_bibliography(self):
        if self.type != 'research-article':
            print("Unimplemented: parsing bibliography on non-research-article")
            self.bibliography = []
            return

        from citation_parser import citation_grammars
        from citation_parser import CitationVisitor
        from parsimonious.exceptions import ParseError
        cv = CitationVisitor()

        def name_split(x):

            # go backwards, finding the last colon or number. then forwards to the next period.
            parts = re.split(r"([:][^:.]*\.|[0-9][^0-9.]*\.)", x)
            parts = [ "".join( parts[:-1] ), parts[-1] ]

            return parts

        t = self.bibString
        # split by years first, to find separate citations
        yspl = re.split("((?:19|20)[0-9]{2}[^\-])", t)
        print("\n".join("--%s"%x for x in yspl))
        bib_entries = []
        for i in range(1, len(yspl), 2):
            # lands on every year.
            if i == 1:
                bib_entries.append( "".join( [
                    yspl[0],    # stub at beginning
                    yspl[1],    # year that was split
                    name_split( yspl[2] )[0]    # split off the name from the bulk of the citation
                ] ).strip() )
                continue

            bib_entries.append("".join([
                name_split(yspl[i-1])[1],  # name of the last one
                yspl[i],  # year that was split
                name_split(yspl[i+1])[0]  # split off the name from the bulk of the citation
            ]).strip())

        print("\n".join("--%s" % x for x in bib_entries))

        errors = 0
        success = 0
        last_success = None

        parsed_bibs = []

        for bib_entry in bib_entries:
            #print("Parsing %s" % bib_entry)


            # this block of code tries to replace ------ at the beginning of the line
            # with the most recent successfull parse
            if last_success is not None:
                a = last_success['authors']
                fa = list( a[0] )
                fa = fa[-1] + ", " + " ".join(fa[:-1])

                a = [ " ".join( x ) for x in a ]

                if len(a) > 1:
                    author_string = ", ".join(
                        [fa] +
                        a[1:-1] +
                        ["and "+ a[-1]]
                    )
                else:
                    author_string = fa

                bib_entry = re.sub("^[-–—]+", author_string, bib_entry)

            # now we throw it into the parser.
            # not everything works, but it does a pretty good job I think, for now.
            for cg in citation_grammars:
                try:
                    p = cg.parse(bib_entry)
                    p = cv.visit(p)
                    p = p[1][0]

                    last_success = p
                    parsed_bibs.append(p)
                    success += 1
                    break
                except ParseError:
                    errors += 1
                    last_success = None

        self.bibliography = parsed_bibs

    def find_bibliography(self):
        if self.type != 'research-article':
            print("Unimplemented: finding bibliography on non-research-article")
            return

        num = 0
        pid = None
        for pi, p in enumerate(self.pages):
            fr = re.findall( "(REFERENCES|References|Literature Cited)", str(p) )
            if not len(fr):
                continue
            num += len(fr)
            pid = pi

        if num > 1:
            raise ParseError("More than one REFERENCES string!")

        if pid is None:
            raise ParseError("No REFERENCE string found...")

        #print("Found bibliography starting on page ", pid+1)

        bibString = []
        for i in range(pid, len(self.pages)):
            if i == pid:
                bibString.append("".join( re.split("(REFERENCES|References|Literature Cited)", str(self.pages[i]))[2:] ))
                continue

            bibString.append(str(self.pages[i]))

        self.bibString = "\n".join( bibString ).strip()

        newptext = str(self.pages[pid]).split("REFERENCES")[0].strip()
        if newptext == "":
            #print("Deleting page", pid+1)
            self.pages.pop()
        else:
            #print("Truncating page", pid+1)
            self.pages[pid] = Page.from_text( newptext )

        #print("Deleting pages ", pid+2, "through", len(self.pages))
        for i in range(len(self.pages) - pid):
            self.pages.pop()

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "\n\n".join( map(str,self.pages) )


    def __getitem__(self, item):
        return self.metadata[item]

def doi2fn(doi):
    fn = "journal-article-%s.txt" % doi.replace("/", "_")
    for d in dirs:
        p = join(d, fn)
        if exists(p):
            return p

    raise Exception("Journal article not found!", doi)

def fn2doi(fn):
    my_name = ".".join(fn.split(".")[:-1])
    doi = my_name.split("-")[-1].replace("_", "/")
    return doi













# JSTOR FILE ITERATOR!







def getname(x):
    x = x.split("/")[-1]
    x = re.sub(r'(\.xml|\.txt)','',x)
    return x

def file_iterator_basic(zipfiles):
    from random import shuffle
    
    all_files = []
    for zf in zipfiles:
        archive = ZipFile(zf, 'r')
        files = archive.namelist()
        names = list(set(getname(x) for x in files))
        
        all_files += [(archive,name) for name in names]
        
    shuffle(all_files)
        
    for archive, name in all_files:
        try:
            yield(
                name.split("-")[-1].replace("_", "/"),
                archive.read("metadata/%s.xml" % name),
                archive.read("ocr/%s.txt" % name).decode('utf8')
            )
        except KeyError: # some very few articles don't have both
            continue
            
def file_iterator(zipfiles):
    for i, (doi, metadata_str, ocr_str) in enumerate(file_iterator_basic(zipfiles)):
        try:
            drep = clean_metadata( doi, metadata_str )

            if debug: print("got meta")

            if drep['type'] != 'research-article':
                continue

            # some types of titles should be immediately ignored
            def title_looks_researchy(lt):
                lt = lt.lower()
                lt = lt.strip()

                for x in ["book review", 'review essay', 'back matter', 'front matter', 'notes for contributors', 'publication received', 'errata:', 'erratum:']:
                    if x in lt:
                        return False

                for x in ["commentary and debate", 'erratum', '']:
                    if x == lt:
                        return False

                return True

            lt = drep['title'].lower()
            if not title_looks_researchy(lt):
                continue

            # Don't process the document if there are no authors
            if not len(drep['authors']):
                continue

            drep['content'] = get_content_string(ocr_str)

            drep['citations'] = []

            # loop through the matching parentheses in the document
            for index, (parenStart, parenContents) in enumerate(getOuterParens(drep['content'])):

                citations = list(citation_iterator(parenContents))
                if not len(citations):
                    continue


                citation = {
                    "citations": citations,
                    "contextLeft": drep['content'][parenStart-400+1:parenStart+1],
                    "contextRight": drep['content'][parenStart + len(parenContents) + 1:parenStart + len(parenContents) + 1 + 100],
                    "where": parenStart
                }


                # cut off any stuff before the first space
                first_break_left = re.search(r"[\s\.!\?]+", citation['contextLeft'])
                if first_break_left is not None:
                    clean_start_left = citation['contextLeft'][first_break_left.end():]
                else:
                    clean_start_left = citation['contextLeft']

                # worry about parentheses on the LHS...
                clean_start_left = clean_start_left.split(")")
                clean_start_left = clean_start_left[-1]
                    
                # cut off any stuff after the last space
                last_break_right = list(re.finditer(r"[\s\.!\?]+", citation['contextRight']))
                if len(last_break_right):
                    clean_end_right = citation['contextRight'][:last_break_right[-1].start()]
                else:
                    clean_end_right = citation['contextRight']
                    
                # we don't want anything more than a sentence
                sentence_left = sent_tokenize(clean_start_left)
                if len(sentence_left):
                    sentence_left = sentence_left[-1]
                else:
                    sentence_left = ""

                sentence_right = sent_tokenize(clean_end_right)[0]
                if len(sentence_right):
                    sentence_right = sentence_right[0]
                else:
                    sentence_right = ""

                # finally, strip the parentheses from the string
                sentence_left = sentence_left[:-1]
                sentence_right = sentence_right[1:]

                # add the thing in context
                full = sentence_left + "<CITATION>" + sentence_right

                citation['contextPure'] = sentence_left
                #print(full)

                drep['citations'].append(citation)
            
            yield doi, drep
            
        except ParseError as e:
            print("parse error...", e.args, doi)