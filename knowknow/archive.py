from collections import defaultdict
from pathlib import Path
import seaborn
import matplotlib.pyplot as plt
import numpy


def save_figure(name):
    global NB_FNAME, NB_DIR
    outdir = NB_DIR.joinpath("figures")
    if not outdir.exists():
        outdir.mkdir()
    print("Saving to '%s'" % outdir)
    plt.savefig(str(outdir.joinpath("%s.png" % name)), bbox_inches="tight")


def comb(x, y):
    a = set(x.split("."))
    b = set(y.split("."))

    return ".".join(sorted(a.union(b)))


# ============================================================
# TODO
save_nametuples = {}


def make_cross(*args, **kwargs):
    global save_nametuples
    from collections import namedtuple

    if len(args):
        assert (type(args[0]) == dict)
        return make_cross(**args[0])

    keys = tuple(sorted(kwargs))

    if keys in save_nametuples:
        my_named_tuple = save_nametuples[keys]
    else:
        my_named_tuple = namedtuple("_".join(keys), keys)
        save_nametuples[keys] = my_named_tuple

    return my_named_tuple(**kwargs)


# =====================================================
cnt_cache = {}


def get_cnt(name, keys=None):
    if keys is None:
        keys = DEFAULT_KEYS

    cnt = {}

    for k in keys:
        if (name, k) in cnt_cache:
            cnt[k] = cnt_cache[(name, k)]
        else:
            varname = "%s ___ %s" % (name, k)

            # print(k)
            this_cnt = defaultdict(int, named_tupelize(dict(load_variable(varname)), k))
            cnt[k] = this_cnt
            cnt_cache[(name, k)] = this_cnt

    avail = get_cnt_keys(name)

    print("Loaded keys: %s" % cnt.keys())
    print("Available keys: %s" % avail)
    return cnt


DEFAULT_KEYS = ['fj.fy', 'fy', 'c', 'c.fy']


def get_cnt_keys(name):
    avail = variable_dir.glob("%s ___ *" % name)
    avail = [x.name for x in avail]
    avail = [x.split("___")[1].strip() for x in avail]
    return avail


def named_tupelize(d, ctype):
    keys = sorted(ctype.split("."))

    def doit(k):
        if type(k) in [tuple, list]:
            return make_cross(dict(zip(keys, k)))
        elif len(keys) == 1:
            return make_cross({keys[0]: k})
        else:
            raise Exception("strange case...")

    return {
        doit(k): v
        for k, v in d.items()
    }


class VariableNotFound(Exception):
    pass


variable_dir = Path(BASEDIR, "variables")

data_files = {
    'sociology-wos': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/5eded795c67d30014e1f3714/?zip='
}


def load_variable(name):
    import pickle
    from collections import defaultdict, Counter

    nsp = name.split("/")
    if len(nsp) == 1:  # fallback to old ways
        nsp = name.split(".")
        collection = nsp[0]
        varname = ".".join(nsp[1:])
        name = "/".join([collection, varname])
    elif len(nsp) == 2:
        collection, varname = nsp
    else:
        raise Exception("idk how to parse this... help")

    if not variable_dir.joinpath(collection).exists():
        print("collection", collection, "does not exist...")
        print("attempting to load from OSF")

        if collection not in data_files:
            raise Exception("no data file logged for '%s'" % collection)

        zip_dest = Path(BASEDIR, "variables", "%s.zip" % collection)
        if not zip_dest.exists():
            download_file(data_files[collection], zip_dest)

        print("Extracting...", str(zip_dest))
        import zipfile
        with zipfile.ZipFile(str(zip_dest), 'r') as zip_ref:
            zip_ref.extractall(str(zip_dest.parent.joinpath(collection)))

    try:
        return pickle.load(variable_dir.joinpath(name).open('rb'))
    except FileNotFoundError:
        raise VariableNotFound(name)


def download_file(url, outfn):
    import requests
    url = str(url)
    outfn = str(outfn)
    Path(outfn).parent.mkdir(exist_ok=True)
    print("Beginning download, ", url)
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(outfn, 'wb') as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=8192)):
                if i % 1000 == 0 and i:
                    print('%0.2f MB downloaded...' % (i * 8192 / 1e6))
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return outfn


# =================================================
