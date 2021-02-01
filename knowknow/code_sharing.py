__all__ = [
    'require'
]

from knowknow.env import GLOBS

modules = {
    'citation-deaths': "G:/My Drive/2020 ORGANISATION/1. PROJECTS/qualitative analysis of literature/110 CITATION ANALYSIS/010 analyses/bundle 101 - citation deaths reboot 10-2020",
    'wos-counter': "G:/My Drive/2020 ORGANISATION/1. PROJECTS/qualitative analysis of literature/110 CITATION ANALYSIS/010 analyses/bundle 102 - creating wos database",
    'stats': "G:/My Drive/2020 ORGANISATION/1. PROJECTS/qualitative analysis of literature/110 CITATION ANALYSIS/010 analyses/bundle 103 - various statistics"
    #'summary-stats': ""
}

#require('amcgail/citation-deaths', '.')

mod_cache = {}

def require(where, what):

    from pathlib import Path

    wsp = where.split("/")
    if len(wsp) == 1:
        who, name = None, wsp[0]
    else:
        who, name = wsp

    dest_fn = Path(GLOBS['kkdir'], 'code', name)

    if not dest_fn.exists():
        print("Module doesn't exist. Attempting to load it...")

        if where[0] == '@':
            where = where[1:]

        url = "https://github.com/%s" % where

        short_name = url.split("/")[-1]

        print("Cloning '%s' into '<kk>/code/%s' ..." % (name, short_name))

        from git.repo.base import Repo
        fld = Path(GLOBS['kkdir'], 'code')
        if not fld.exists():
            fld.mkdir()
        Repo.clone_from(url, fld.joinpath(short_name))


    wsp = what.split("/")
    if len(wsp) == 1:
        # get it from __init__
        modfn = dest_fn.joinpath("__init__.py")

    elif len(wsp) == 2:
        # get it from file within
        modfn = dest_fn.joinpath("%s.py" % wsp[0])
    else:
        raise Exception('yolo')


    if modfn in mod_cache:
        mod = mod_cache[modfn]
    else:
        from runpy import run_path
        mod = run_path( modfn )

        if '__all__' in mod:
            mod = {
                k: v for k,v in mod.items() if k in mod['__all__']
            }

        mod_cache[modfn] = mod

    return mod[wsp[-1]]