import re

def fix_paragraph(x):
    # replace what look like hyphenations
    x = re.sub(r"(\S+)-[\n\r]\s*(\S+)", r"\1\2", x)

    # replace newlines with spaces
    x = re.sub(r"[\n\r]", " ", x)

    # replace multiple spaces
    x = re.sub(r" +", " ", x)

    return x

def follow_expand( from_what, follow_what ):
    """
    A SpaCy dependency graph follower...
    :param from_what:
    :param follow_what:
    :return:
    """
    found = [x for x in from_what.children if x.dep_ == follow_what]
    return [
        " ".join(str(x) for x in one.subtree)
        for one in found
    ]

def untokenize(tokens):
    import string
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def getname(x):
    x = x.split("/")[-1]
    x = re.sub(r'(\.xml|\.txt)', '', x)
    return x



# The following methods are used for extracting in-text citations from the body of a document...

import re
import calendar

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
    r"^{prefix}{names}, {year}{postfix}$".format(**locals()),
    r"^{prefix}{names} {year}, {ppnum}{postfix}$".format(**locals()),
    r"^{prefix}{names} {year}{postfix}$".format(**locals()),
    r"^{prefix}{names}, {year}: {pnum}{postfix}$".format(**locals()),
    r"^{prefix}{names} \[{year}\]{postfix}$".format(**locals()),
    r"^{prefix}{names} \[{year}, [^\]]\]{postfix}$".format(**locals()),
]

# print(regex_in_order)

name_blacklist = calendar.month_name[1:]

faulty_years = set()


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

            if paren_level == 2:  # hack to remove multiple levels
                paren_level = 1

        if c == ")":
            paren_level -= 1

            if paren_level <= 0:
                rets.append((i_start, buffer[1:]))

                paren_level = 0  # hack to remove multiple levels

                buffer = ""
        if paren_level > 0:
            buffer += c

    return rets


def extractCitation(t):
    t = re.sub(r"[-]\n", "", t)
    t = re.sub(r"\s+", " ", t)

    # try splitting it both ways,
    # just see which gets more hits

    tparts = t.split(";")
    # print("A")
    ret = []
    for tpart in tparts:
        tpart = tpart.strip()
        # print("Aa")
        # I guess this causes infinite looping??
        if len(tpart.split(",")) >= 4:
            continue
        # print("Ab")
        # print("Checking", tpart)
        for rgxi, rgx in enumerate(regex_in_order):
            mtch = re.match(rgx, tpart)
            if mtch:
                d = mtch.groupdict()
                if d['names'] in name_blacklist:
                    continue
                ret.append(d)
                break  # wow... didn't have this before and was getting annoying duplicates smh...
    # print("B")
    tparts2 = t.split(",")

    ret2 = []
    for tpart in tparts2:
        tpart = tpart.strip()

        # print("Checking", tpart)
        for rgx in regex_in_order:
            mtch = re.match(rgx, tpart)
            if mtch:
                d = mtch.groupdict()
                if d['names'] in name_blacklist:
                    continue
                ret2.append(d)
    # print("C")
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