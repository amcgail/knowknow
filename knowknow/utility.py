"""
Utilities
=======================================
Functions that don't need state of knowknow engine.
Mostly are meant to generate empty/autofilled data structures or for type transformation.
"""
from collections import namedtuple
from pathlib import Path
import requests
from logzero import logger


def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if tuple(args) in cache:
            return cache[tuple(args)]
        result = func(*args)
        cache[tuple(args)] = result
        return result

    return memoized_func


def comb(x, y):
    """
    TODO: figure out
    :param x: . separated values
    :param y: . separated values
    :return: union of unique values from x and y. . separated
    'q.w.e', 'e.r.t' -> 'e.q.r.t.w'
    """
    a = set(x.split("."))
    b = set(y.split("."))

    return ".".join(sorted(a.union(b)))


def make_cross_kwargs(*args, **kwargs):
    """
    Looks like it turns a dictionary into a named tuple.
    TODO: get rid of kwargs logic. Why generating a named tuple is named cross?
    :param args:
    :param kwargs:
    :return:
    """
    # removed caching
    if len(args):
        assert (type(args[0]) == dict)
        return make_cross_kwargs(**args[0])

    keys = tuple(sorted(kwargs))

    my_named_tuple = namedtuple("_".join(keys), keys)

    return my_named_tuple(**kwargs)


@memoize
def gen_tuple_template(keys: tuple):
    """
    :param keys: a tuple of strings
    :return: a named tuple template
    ('e', 'q', 'w') -> <class '__main__.e_q_w'>
    """
    # TODO: add caching
    return namedtuple('_'.join(keys), keys)


def make_cross(key___value: dict):
    """
    :param key___value:
    :return:
    {'1': 'q', '2': 'w', '3': 'e'} -> e_q_w(e='3', q='1', w='2')
    """
    return gen_tuple_template(tuple(sorted(key___value.keys())))(**key___value)


@memoize
def doit(k, keys: tuple):
    if type(k) in [tuple, list]:
        return make_cross(dict(zip(keys, k)))
    elif len(keys) == 1:
        # TODO: where this hack is used?
        # logger.warning(f'DoIt function in named_tupelize uses len(1) hack. ctype: {ctype}, '
        #                f'd: {d}, curr_key {k}')
        return make_cross({keys[0]: k})
    else:
        raise Exception(f"Cannot transform {k} into a named tuple with keys {keys}")


def named_tupelize(d: dict, ctype: str):
    """
    For each key in d, try making it a named tuple. Return {namedtuple(k, ctypes): v for k, v in dict)
    :param d:
    :param ctype: category types?
    :return:
    """
    keyss = tuple(sorted(ctype.split(".")))
    return {doit(k, keyss): v for k, v in d.items()}


class VariableNotFound(Exception):
    pass


def download_file(url, outfn):
    url = str(url)
    outfn = str(outfn)
    Path(outfn).parent.mkdir(exist_ok=True)
    logger.info(f"Beginning download, {url}")
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(outfn, 'wb') as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=8192)):
                if i % 1000 == 0 and i:
                    logger.info('%0.2f MB downloaded...' % (i * 8192 / 1e6))
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return outfn
