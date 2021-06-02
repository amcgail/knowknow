from .. import *
from ..nlp import *

#from jstor_xml import *
#from intext import *

from nltk import sent_tokenize
from lxml.etree import _ElementTree as ElementTree
from lxml import etree
recovering_parser = etree.XMLParser(recover=True)

from zipfile import ZipFile





if False:
    # looks like I don't use this?
    import spacy

    nlp = spacy.load('en')
    import neuralcoref

    neuralcoref.add_to_pipe(nlp)





ALL_UNIVERSITIES = {
    x[1].upper().strip()
    for x in creader(
        Path(__file__).parent.parent.joinpath("external-data", "world-universities.csv")
            .open('r', encoding='UTF8')
    )
}
max_uni_length = max( len(x) for x in ALL_UNIVERSITIES )







def citation_iterator(t):
    citations = extractCitation(t)
    for c in citations:
        name = c['names']
        name = name.replace("'s", "")  # change "Simmel's" to "Simmel"

        years = c['year'].split(",")
        for y in years:
            y = y.strip()  # get rid of the extra spacing
            y = re.findall(r'[0-9]{4}', y)  # this strips out the 'a' in '1995a'
            if not len(y):
                faulty_years.add(c['year'])
                continue

            y = y[0]
            yield "%s (%s)" % (name, y)


class ParseError(Exception):
    pass

def clean_metadata( doi, metadata_str ):

    try:
        metaXml = etree.fromstring(metadata_str, parser=recovering_parser)
    except OSError:
        raise ParseError("NO METADATA!", doi)

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

romanChars = "LIXMCV"

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
    def from_xml_strings(cls, meta_xml, content_xml, doi=None, complex_parsing=True):

        metadata = cls.parse_metadata(meta_xml)
        metadata['doi'] = doi
        page_strings = cls.parse_content(content_xml)

        page_strings = list(map(basic_ocr_cleaning, page_strings))
        page_strings = [x for x in page_strings if x != ""]
        if not len(page_strings):
            raise ParseError("Empty document")

        return Document.from_pages(page_strings, metadata=metadata, complex_parsing=complex_parsing)


    @classmethod
    def from_file(cls, fn, complex_parsing=True):

        my_name = ".".join(fn.split(".")[:-1])
        doi = my_name.split("-")[-1].replace("_", "/")

        metadataFn = "%s.xml" % my_name
        metadataFn = join(dirname(metadataFn), "..", "metadata", metadataFn)

        metadata = cls.parse_metadata(open(metadataFn).read())


        page_strings = cls.parse_content(open(fn).read())

        page_strings = list(map(basic_ocr_cleaning, page_strings))
        page_strings = [x for x in page_strings if x != ""]
        if not len(page_strings):
            raise ParseError("Empty document")

        return Document.from_pages(page_strings, metadata=metadata, complex_parsing=complex_parsing)

    @classmethod
    def from_pages(cls, page_strings, metadata={}, complex_parsing=True):
        d = Document(metadata)

        if not len(page_strings):
            raise ParseError("No pages...")

        d.pages = [Page.from_lines(re.split(r"[\n\r]+", page)) for page in page_strings]

        if complex_parsing:
            if d.type == 'research-article':
                try:
                    d.find_bibliography()
                    d.extract_headers_footers()
                    #d.parse_bibliography()
                except ParseError:
                    print("Couldn't extract bib and headers from ", d['doi'])
                    raise

        return d

    @property
    def sentences(self):
        for i, p in enumerate(self.pages):
            if i > 0:
                yield Sentence(words=self.pages[i - 1].end_stub + self.pages[i].start_stub)

            p.extract_sentences()
            for s in p.full_sentences:
                yield s

    @classmethod
    def parse_content(cls, xml_string):

        import xml.etree.ElementTree as ET
        docXml = ET.ElementTree(ET.fromstring(xml_string, parser=recovering_parser))

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
                # print(try_another)
                if try_another == '':
                    continue

                page_strings.append(try_another)
            else:
                page_strings.append(s.text.strip())

        return page_strings

    @classmethod
    def parse_metadata(cls, xml_string):

        try:
            import xml.etree.ElementTree as ET
            metaXml = ET.ElementTree(ET.fromstring(xml_string, parser=recovering_parser))
        except OSError:
            raise ParseError("NO METADATA!", fn)

        assert (isinstance(metaXml, ET.ElementTree))

        def findAndCombine(query):
            return ";".join([" ".join(x.itertext()) for x in metaXml.findall(query)]).strip()

        metadata = {}
        metadata['type'] = metaXml.getroot().get("article-type")

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
            print('no authors...')
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
            p.page_mapped = num_pdf2page(pdf_num=i, page_map=self.page_map)

    def eliminate_metadata(self):
        from fuzzysearch import find_near_matches

        if self.type != 'research-article':
            print("Unimplemented: extracting headers and footers on non-research-article")
            return

        if not len(self.pages):
            raise ParseError()

        fp = str(self.pages[0])
        fp = fp.upper()

        # extracting title, heading stuff on first page, and abstract
        to_extract_from_first_page = [
            self['title']
        ]

        if len(self['abstract']):
            to_extract_from_first_page.append(self['abstract'])

        for a in self['authors']:
            arev = a.split()
            arev = arev[-1] + ", " + " ".join(arev[:-1])

            to_extract_from_first_page += [a, arev]

        found = []

        # to get the university, not included in most metadatas...
        if False:
            found_uni = []
            for u in ALL_UNIVERSITIES:
                found.append((fp.find(u), len(u)))
        print(to_extract_from_first_page)

        for word in to_extract_from_first_page:
            for match in find_near_matches(word.upper(), fp, max_l_dist=len(word) // 10):
                # match = word[match.start:match.end]
                # print('match {}'.format(match))
                # index = ls.find(match)
                print(match.start, match.end, match.matched, match.dist)
                found.append( (match.start, match.end-match.start) )



        # limit to those strings which were actually found
        found = list(filter(lambda x: x[0] >= 0, found))

        if len(found):
            # these strings should be "end to end", i.e. less than 5 characters separating them
            found_ends = [x[0] + x[1] for x in found]
            found_min_beginning = min(x[0] for x in found)
            found = list(
                filter(lambda x: any(abs(x[0] - y) < 5 for y in found_ends) or x[0] == found_min_beginning, found)
            )

            # we then cut after these metadata fields!
            doc_start = max(found, key=lambda x: x[0] + x[1])
            doc_start = doc_start[0] + doc_start[1]

            newfp_text = str(self.pages[0])[doc_start:].strip()
            # newfp = Page.from_text(newfp_text)
            self.pages[0] = newfp_text
            #print("Trimmed woohoo: %s\n\n%s"%(fp, newfp_text))

    def extract_headers_footers(self):
        # extracting headers via voting
        num_identical_necessary = len(self.pages) / 4

        flines = [str(x).split()[:20] for x in self.pages]
        grams = Counter(
            tuple(x[:i])
            for i in range(1, 15)
            for x in flines
        )

        candidates = set(x for x, c in grams.items() if c > num_identical_necessary)
        maximal_candidates = []
        for c in candidates:
            keep = True
            for c2 in candidates:
                if len(c) >= len(c2):
                    continue
                if c[:min(len(c), len(c2))] == c2[:min(len(c), len(c2))]:
                    keep = False

            if keep:
                maximal_candidates.append(c)

        self.headers = [" ".join(x) for x in maximal_candidates]
        for h in self.headers:

            # don't want to get rid of the title on the first page.
            # that's the job for another algorithm
            # therefore, the [1:]
            for i, p in enumerate(self.pages[1:]):
                try:
                    if str(p).index(h) < 30:
                        self.pages[i+1] = Page.from_text(str(p).replace(h, "").strip())
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
            parts = ["".join(parts[:-1]), parts[-1]]

            return parts

        t = self.bibString
        # split by years first, to find separate citations
        yspl = re.split("((?:19|20)[0-9]{2}[^\-])", t)
        print("\n".join("--%s" % x for x in yspl))
        bib_entries = []
        for i in range(1, len(yspl), 2):
            # lands on every year.
            if i == 1:
                bib_entries.append("".join([
                    yspl[0],  # stub at beginning
                    yspl[1],  # year that was split
                    name_split(yspl[2])[0]  # split off the name from the bulk of the citation
                ]).strip())
                continue

            bib_entries.append("".join([
                name_split(yspl[i - 1])[1],  # name of the last one
                yspl[i],  # year that was split
                name_split(yspl[i + 1])[0]  # split off the name from the bulk of the citation
            ]).strip())

        print("\n".join("--%s" % x for x in bib_entries))

        errors = 0
        success = 0
        last_success = None

        parsed_bibs = []

        for bib_entry in bib_entries:
            # print("Parsing %s" % bib_entry)

            # this block of code tries to replace ------ at the beginning of the line
            # with the most recent successfull parse
            if last_success is not None:
                a = last_success['authors']
                fa = list(a[0])
                fa = fa[-1] + ", " + " ".join(fa[:-1])

                a = [" ".join(x) for x in a]

                if len(a) > 1:
                    author_string = ", ".join(
                        [fa] +
                        a[1:-1] +
                        ["and " + a[-1]]
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
            fr = re.findall("(REFERENCES|References|Literature Cited)", str(p))
            if not len(fr):
                continue
            num += len(fr)
            pid = pi

        if num > 1:
            #raise ParseError("More than one REFERENCES string!")
            self.bibString = ""
            return

        if pid is None:
            #raise ParseError("No REFERENCE string found...", "\n".join( str(p) for p in self.pages ))
            self.bibString = ""
            return

        # print("Found bibliography starting on page ", pid+1)

        bibString = []
        for i in range(pid, len(self.pages)):
            if i == pid:
                bibString.append("".join(re.split("(REFERENCES|References|Literature Cited)", str(self.pages[i]))[2:]))
                continue

            bibString.append(str(self.pages[i]))

        self.bibString = "\n".join(bibString).strip()

        newptext = str(self.pages[pid]).split("REFERENCES")[0].strip()
        if newptext == "":
            # print("Deleting page", pid+1)
            self.pages.pop()
        else:
            # print("Truncating page", pid+1)
            self.pages[pid] = Page.from_text(newptext)

        # print("Deleting pages ", pid+2, "through", len(self.pages))
        for i in range(len(self.pages) - pid):
            self.pages.pop()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "\n\n".join(map(str, self.pages))

    def __getitem__(self, item):
        return self.metadata[item]


_zp_cache = {}


def doi2xml(zipdir, doi):
    zf, fn, fn2 = doi2fn(zipdir, doi)

    archive = ZipFile(zf, 'r')
    return (
        archive.read(fn),
        archive.read(fn2)
    )

def load_zd(zipdir):
        zipfiles = list(Path(zipdir).glob("*.zip"))

        zret = defaultdict(set)

        for zf in zipfiles:
            archive = ZipFile(zf, 'r')
            files = archive.namelist()

            for fn in files:
                if 'ocr/' in fn or 'ngram/' in fn:
                    continue

                fn2 = fn.replace("metadata", "ocr").replace(".xml", ".txt")

                mydoi = fn2doi(fn)
                zret[ mydoi ] = (zf, fn, fn2)

        _zp_cache[zipdir] = zret

def doi2fn(zipdir, doi):
    fn = "journal-article-%s.txt" % doi.replace("/", "_")

    if zipdir not in _zp_cache:
        load_zd(zipdir)

    return _zp_cache[zipdir][ doi ]


def fn2doi(fn):
    my_name = ".".join(fn.split(".")[:-1])
    doi = my_name.split("-")[-1].replace("_", "/")
    return doi












def _zip_doc_iterator(zipfiles):
    from random import shuffle

    all_files = []
    for zf in zipfiles:
        archive = ZipFile(zf, 'r')
        files = archive.namelist()
        names = list(set(getname(x) for x in files))

        all_files += [(archive, name) for name in names]

    shuffle(all_files)

    for archive, name in all_files:
        try:
            yield (
                name.split("-")[-1].replace("_", "/"),
                archive.read("metadata/%s.xml" % name),
                archive.read("ocr/%s.txt" % name).decode('utf8')
            )
        except KeyError:  # some very few articles don't have both
            continue




# some types of titles should be immediately ignored
def title_looks_researchy(lt):
    lt = lt.lower()
    lt = lt.strip()

    for x in ["book review", 'review essay', 'back matter', 'front matter', 'notes for contributors',
              'publication received', 'errata:', 'erratum:']:
        if x in lt:
            return False

    for x in ["commentary and debate", 'erratum', '']:
        if x == lt:
            return False

    return True



def doc_iterator(jstor_zip_base, research_limit=True, complex_parsing=True):
    zipfiles = list(Path(jstor_zip_base).glob("*.zip"))

    for i, (doi, metadata_str, ocr_str) in enumerate(_zip_doc_iterator(zipfiles)):

        try:
            drep = Document.parse_metadata(metadata_str)

            if research_limit:
                if drep['type'] != 'research-article':
                    continue

                lt = drep['title'].lower()
                if not title_looks_researchy(lt):
                    continue

                # Don't process the document if there are no authors
                if not len(drep['authors']):
                    continue

            d = Document.from_xml_strings(metadata_str, ocr_str, complex_parsing=complex_parsing)
            d.metadata['doi'] = doi
        except ParseError:
            continue

        yield d


def doc_iterator_forMongo(jstor_zip_base,
                 journals_filter=None,
                 debug=False):

    zipfiles = list(Path(jstor_zip_base).glob("*.zip"))

    for i, (doi, metadata_str, ocr_str) in enumerate(_zip_doc_iterator(zipfiles)):
        try:
            drep = clean_metadata(doi, metadata_str)

            # only include journals in the list "included_journals"
            if journals_filter is not None and (drep['journal'].lower() not in journals_filter):
                continue

            if debug: print("got meta")

            if drep['type'] != 'research-article':
                continue

            # some types of titles should be immediately ignored
            def title_looks_researchy(lt):
                lt = lt.lower()
                lt = lt.strip()

                for x in ["book review", 'review essay', 'back matter', 'front matter', 'notes for contributors',
                          'publication received', 'errata:', 'erratum:']:
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
                    "contextLeft": drep['content'][parenStart - 400 + 1:parenStart + 1],
                    "contextRight": drep['content'][
                                    parenStart + len(parenContents) + 1:parenStart + len(parenContents) + 1 + 100],
                    "where": parenStart
                }

                # cut off any stuff before the first space
                first_break_left = re.search(r"[\s.!?]+", citation['contextLeft'])
                if first_break_left is not None:
                    clean_start_left = citation['contextLeft'][first_break_left.end():]
                else:
                    clean_start_left = citation['contextLeft']

                # cut off any stuff after the last space
                last_break_right = list(re.finditer(r"[\s.!?]+", citation['contextRight']))
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
                # print(full)

                drep['citations'].append(citation)

            yield doi, drep

        except ParseError as e:
            print("parse error...", e.args, doi)










class jstor_counter:

    def __init__(
            self,
            jstor_zip_base, output_database, name_blacklist=[],
            RUN_EVERYTHING=False,
            groups=None, group_reps=None,
            citations_filter=None, journals_filter=None, debug=False,
            CONSOLIDATE_ITERS=0, term_whitelist = set(), NUM_TERMS_TO_KEEP=100, stopwords = None
    ):

        if stopwords is None:
            # getting ready for term counting
            from nltk.corpus import stopwords as sw
            self.stopwords = set(sw.words('english'))
        else:
            self.stopwords = stopwords

        self.cits = 0
        self.last_print = 0
        self.citations_skipped = 0

        self.jstor_zip_base = Path(jstor_zip_base)
        assert(self.jstor_zip_base.exists())

        self.output_database = output_database
        self.name_blacklist = name_blacklist
        self.RUN_EVERYTHING = RUN_EVERYTHING
        self.groups = groups
        self.group_reps = group_reps
        self.citations_filter = citations_filter
        self.journals_filter = journals_filter
        self.debug = debug

        self.CONSOLIDATE_ITERS = CONSOLIDATE_ITERS
        self.NUM_TERMS_TO_KEEP = NUM_TERMS_TO_KEEP

        # Instantiating counters
        self.ind = defaultdict(lambda: defaultdict(int))
        self.track_doc = defaultdict(lambda: defaultdict(set))
        self.doc = defaultdict(lambda: defaultdict(int))

        self.term_whitelist = term_whitelist





    def cnt(self, term, space, doc):
        if ".".join(sorted(space.split("."))) != space:
            raise Exception(space, "should be sorted...")

        # it's a set, yo
        self.track_doc[space][term].add(doc)
        # update cnt_doc
        self.doc[space][term] = len(self.track_doc[space][term])
        # update ind count
        self.ind[space][term] += 1





    def consolidate_terms(self):
        have_now = set(self.doc['t'])
        # this is where the filtering occurs

        to_keep = set()
        if True:
            # takes terms based on the maximum number I can take...
            terms = list(self.doc['t'].keys())
            counts = np.array([self.doc['t'][k] for k in terms])
            argst = list(reversed(np.argsort(counts)))

            to_keep = [terms[i] for i in argst if '-' in terms[i][0]][
                      :self.NUM_TERMS_TO_KEEP // 2]  # half should be 2-tuples
            to_keep += [terms[i] for i in argst if not '-' in terms[i][0]][
                       :self.NUM_TERMS_TO_KEEP // 2]  # half should be 1-tuples

            to_remove = have_now.difference(to_keep)
            to_remove = set("-".join(x) for x in to_remove)

        if False:
            # takes the top 5000 terms in terms of yearly count
            sort_them = sorted(self.doc['fy.t'], key=lambda x: -self.doc['fy.t'][x])
            to_keep = defaultdict(set)

            i = 0
            while not len(to_keep) or (
                    min(len(x) for x in to_keep.values()) < NPERYEAR and
                    i < len(sort_them)
            ):
                # adds the term to the year set, if it's not already "full"
                me = sort_them[i]
                me_fy, me_t = me

                # eventually, we don't count it :P
                if self.doc['t'][me_t] < CONSOLIDATION_CUTOFF:
                    break

                if len(to_keep[me_fy]) < NPERYEAR:
                    to_keep[me_fy].add(me_t)
                i += 1

            if False:  # useful for debugging
                print({
                    k: len(v)
                    for k, v in to_keep.items()
                })

            to_keep = set(chain.from_iterable(x for x in to_keep.values()))
            to_remove = have_now.difference(to_keep)

        # so that we never log counts for these again:
        self.term_whitelist.update([x[0] for x in to_keep])

        # the rest of the code is pruning all other term counts for this term in memory
        print("consolidating... removing", len(to_remove), 'e.g.', sample(to_remove, 5))

        to_prune = ['t', 'fy.t', 'fj.t', 'c.t']
        for tp in to_prune:

            whichT = tp.split(".").index('t')  # this checks where 't' is in the name of the variable (first or second?)

            print("pruning '%s'..." % tp)

            tydels = [x for x in self.doc[tp] if x[whichT] in to_remove]

            print("old size:", len(self.doc[tp]))
            for tr in tydels:
                del self.doc[tp][tr]
                del self.ind[tp][tr]
            print("new size:", len(self.doc[tp]))

        print("final terms: ", ", ".join(sample(list("-".join(list(x)) for x in self.doc['t']), 200)))


    def account_for(self, doc):

        # consolidating "terms" counter as I go, to limit RAM overhead
        # I'm only interested in the most common 1000
        if self.CONSOLIDATE_ITERS > 0:
            if self.cits - self.last_print > self.CONSOLIDATE_ITERS:
                print("Citation %s" % self.cits)
                print("Term %s" % len(self.doc['t']))
                # print(sample(list(self.doc['t']), 10))
                last_print = self.cits
                self.consolidate_terms()

        if 'citations' not in doc or not len(doc['citations']):
            # print("No citations", doc['doi'])
            return

        for c in doc['citations']:
            if 'contextPure' not in c:
                raise Exception("no contextPure...")

            for cited in c['citations']:

                if self.citations_filter is not None and (cited not in self.citations_filter):
                    citations_skipped += 1
                    continue

                self.cits += 1
                self.cnt(doc['year'], 'fy', doc['doi'])

                # citation
                self.cnt(cited, 'c', doc['doi'])

                # journal
                self.cnt(doc['journal'], 'fj', doc['doi'])

                # journal year
                self.cnt((doc['journal'], doc['year']), 'fj.fy', doc['doi'])

                # citation journal
                self.cnt((cited, doc['journal']), 'c.fj', doc['doi'])

                # citation year
                self.cnt((cited, doc['year']), 'c.fy', doc['doi'])

            # constructing the tuples set :)
            sp = c['contextPure'].lower()
            sp = re.sub("[^a-zA-Z\s]+", "", sp)  # removing extraneous characters
            sp = re.sub("\s+", " ", sp)  # removing extra characters
            sp = sp.strip()
            sp = sp.split()  # splitting into words

            sp = [x for x in sp if x not in self.stopwords]  # strip stopwords

            if False:
                tups = set(zip(sp[:-1], sp[1:]))  # two-word tuples
            elif False:
                tups = set((t1, t2) for t1 in sp for t2 in sp if t1 != t2)  # every two-word pair :)
            else:

                tups = set("-".join(sorted(x)) for x in set(zip(sp[:-1], sp[1:])))  # two-word tuples
                tups.update(sp)  # one-word tuples

            # print(len(tups),c['contextPure'], "---", tups)

            if len(self.term_whitelist):
                tups = [x for x in tups if x in term_whitelist]

            # just term count, in case we are using the `basic` mode
            for t1 in tups:
                # term
                self.cnt((t1,), 't', doc['doi'])

                # term year
                self.cnt((doc['year'], t1), 'fy.t', doc['doi'])

            if self.RUN_EVERYTHING:

                for cited in c['citations']:

                    if use_included_citations_filter and (cited not in included_citations):
                        continue

                    # term features
                    for t1 in tups:

                        # cited work, tuple
                        self.cnt((cited, t1), 'c.t', doc['doi'])

                        # term journal
                        self.cnt((doc['journal'], t1), 'fj.t', doc['doi'])

                        if False:  # eliminating data I'm not using

                            # author loop
                            for a in doc['authors']:
                                # term author
                                self.cnt((a, t1), 'fa.t', doc['doi'])

                        if len(self.term_whitelist):  # really don't want to do this too early. wait until it's narrowed down to the 5k
                            # term term...
                            for t2 in tups:
                                # if they intersect each other, continue...
                                if len(set(t1).intersection(set(t2))) >= min(len(t1), len(t2)):
                                    continue

                                # term term
                                self.cnt((t1, t2), 't.t', doc['doi'])

                    # author loop
                    for a in doc['authors']:
                        # citation author
                        self.cnt((cited, a), 'c.fa', doc['doi'])

                        # year author journal
                        self.cnt((a, doc['journal'], doc['year']), 'fa.fj.fy', doc['doi'])

                        # author
                        self.cnt((a,), 'fa', doc['doi'])

                    # add to counters for citation-citation counts
                    for cited1 in c['citations']:
                        for cited2 in c['citations']:
                            if cited1 >= cited2:
                                continue

                            self.cnt((cited1, cited2), 'c.c', doc['doi'])
                            self.cnt((cited1, cited2, doc['year']), 'c.c.fy', doc['doi'])


    def count(self):

        seen = set()

        skipped = 0

        total_count = Counter()
        doc_count = Counter()
        pair_count = Counter()

        debug = False

        for i, (doi, drep) in enumerate(self.doc_iterator()): #!!!!!!!!!!!!!!!!! fix this now...

            if i % 1000 == 0:
                print("Document", i, "...",
                      len(self.doc['fj'].keys()), "journals...",
                      len(self.doc['c'].keys()), "cited works...",
                      len(self.doc['fa'].keys()), "authors...",
                      len(self.doc['t'].keys()), "terms used...",
                      #citations_skipped, "skipped citations...",
                      self.doc['t'][('social',)], "'social' terms"
                      )

            # sometimes multiple journal names map onto the same journal, for all intents and purposes
            #if drep['journal'] in self.journjournal_map:
            #    drep['journal'] = journal_map[drep['journal']]

                # only include journals in the list "included_journals"
            if self.journals_filter is not None and (drep['journal'] not in self.journals_filter):
                continue

            # now that we have all the information we need,
            # we simply need to "count" this document in a few different ways
            self.account_for(drep)

    def save_counters(self):
        db = Dataset(self.output_database)
        for k, count in self.doc.items():
            varname = "doc ___ %s" % k
            db.save_variable(varname, dict(count))

        for k, count in self.ind.items():
            varname = "ind ___ %s" % k
            db.save_variable(varname, dict(count))