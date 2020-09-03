import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import seaborn
import knowknow




class Constants:

    DEFAULT_KEYS = ['fj.fy', 'fy']  # , 'c', 'c.fy']
    data_files = {
        'sociology-wos': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/'
                         '5eded795c67d30014e1f3714/?zip=',
        'sociology-jstor': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/'
                           '5eded76fc67d3001491f27d1/?zip=',
        'sociology-jstor-basical': 'https://files.osf.io/v1/resources/9vx4y/providers/osfstorage/'
                                   '5eded7478b542601748b4bdb/?zip='
    }


class KnowKnow:

    def __init__(self, NB_DIR=None, BASEDIR=None):
        '''
        Initailize KnowKnow class object
        :param NB_DIR:
        :param BASEDIR: base directory path
        '''
        self.NB_DIR = NB_DIR
        self.BASEDIR = BASEDIR

        self.variable_dir = Path(self.BASEDIR, 'variables')
        self.BASEDIR = Path(self.BASEDIR)
        os.makedirs(self.variable_dir, exist_ok=True)

    def save_figure(self, name):
        outdir = os.path.join(self.BASEDIR, 'figures')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print("Saving to '%s'" % outdir)
        plt.savefig(str(os.path.join(outdir, "%s.png" % name)), bbox_inches="tight")

    def get_cnt_keys(self, name):
        avail = self.variable_dir.glob("%s ___ *" % name)
        avail = [x.name for x in avail]
        avail = [x.split("___")[1].strip() for x in avail]
        return avail

    def get_cnt(self, name, keys=None):
        # TODO: add caching
        if keys is None:
            keys = Constants.DEFAULT_KEYS

        cnt = {}

        for k in keys:
            varname = "%s ___ %s" % (name, k)

            # print(k)
            this_cnt = defaultdict(int, knowknow.named_tupelize(dict(self.load_variable(varname)), k))
            cnt[k] = this_cnt

        avail = self.get_cnt_keys(name)

        print("Loaded keys: %s" % cnt.keys())
        print("Available keys: %s" % avail)
        return cnt

    def load_variable(self, name):
        import pickle

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

        if not self.variable_dir.joinpath(collection).exists():
            print("collection", collection, "does not exist...")
            print("attempting to load from OSF")

            if collection not in Constants.data_files:
                raise Exception("no data file logged for '%s'" % collection)

            zip_dest = Path(self.BASEDIR, "variables", "%s.zip" % collection)
            if not zip_dest.exists():
                knowknow.download_file(Constants.data_files[collection], zip_dest)

            print("Extracting...", str(zip_dest))
            import zipfile
            with zipfile.ZipFile(str(zip_dest), 'r') as zip_ref:
                zip_ref.extractall(str(zip_dest.parent.joinpath(collection)))

        try:
            return pickle.load(self.variable_dir.joinpath(name).open('rb'))
        except FileNotFoundError:
            raise knowknow.VariableNotFound(name)

    def badass_heatmap(self, whats, cnt, dtype, ysum, ify, fnargs=[], RANGE=None,
                       markers={}, markersize=50, align='left',
                       proportional=False, MAXYEAR=2018,
                       database_name=None, **kwargs):
        whats = list(whats)

        all_years = []

        if RANGE is None:
            RANGE = 2015 - min(x.fy for x in cnt[ify] if (getattr(x, dtype),) in whats)

        max_cnt = max([c for x, c in cnt[ify].items() if c > 0 and (getattr(x, dtype),) in whats])

        start_years = []

        if align == 'left':
            for what in whats:
                what = what[0]  # get the item out of the tuple

                if what not in ysum:
                    continue

                start_year = min([x.fy for x in cnt[ify] if cnt[ify][x] > 0 and getattr(x, dtype) == what])

                def get_val(y):

                    nanval = -max_cnt / 5
                    # if what in markers and y in markers[what]:
                    #    return nanval

                    if y < start_year or y > MAXYEAR:
                        return nanval

                    myiy = knowknow.make_cross({"fy": y, dtype: what})
                    return cnt[ify][myiy]

                year_l = [get_val(y) for y in range(start_year, start_year + RANGE)]
                all_years.append(year_l)
                start_years.append(start_year)

        elif align == 'right':
            for what in whats:
                what = what[0]  # get the item out of the tuple

                if what not in ysum:
                    continue

                start_year = MAXYEAR - RANGE

                def get_val(y):

                    nanval = -max_cnt / 5
                    # if what in markers and y in markers[what]:
                    #    return nanval

                    if y < start_year or y > MAXYEAR:
                        return nanval

                    myiy = knowknow.make_cross({"fy": y, dtype: what})
                    return cnt[ify][myiy]

                year_l = [get_val(y) for y in range(start_year, start_year + RANGE)]
                all_years.append(year_l)
                start_years.append(start_year)

        all_years = numpy.array(all_years)

        if proportional is not None:
            if proportional == 'columns':
                all_years = all_years / all_years.sum(axis=0)[None, :]
            if proportional == 'rows':
                all_years = all_years / all_years.sum(axis=1)[:, None]

        # fig, ax = plt.subplots(figsize=(30,10))
        # seaborn.heatmap(all_years, ax=ax)

        # sorts by their closest neighbors
        # very slow line
        distances = numpy.array([
            [
                numpy.sum(numpy.abs(year1[i] - year2[i]) if (year1[i] != -10 and year2[i] != -10) else -10 for i in
                          range(year1.shape[0]))
                for year2 in all_years
            ]
            for year1 in all_years
        ])

        seq = [0]
        while len(seq) < all_years.shape[0]:
            last_one = seq[-1]
            which_done = numpy.array([samp in seq for samp in range(all_years.shape[0])])

            minv = None
            mini = None
            for i in range(distances.shape[0]):
                if i in seq:
                    continue

                v = distances[i, last_one]
                if minv is None or v < minv:
                    mini = i
                    minv = v

            seq.append(mini)

        fig, ax = plt.subplots(figsize=(30, 10))
        seaborn.heatmap(all_years[seq,], ax=ax, **kwargs)

        mx = []
        my = []
        mstyle = []

        for wi, (what, years) in enumerate(markers.items()):
            which_what = whats.index((what,))
            my_start = start_years[which_what]
            which_row = seq.index(which_what)

            for year in years:
                mx.append(year - my_start + 0.5)
                my.append(which_row + 0.5)
                mstyle.append(years[year])  # style!

        # print(markers, mx, my)
        if len(mx):
            for x, y, style in zip(mx, my, mstyle):
                ax.scatter([x], [y], color='black', s=markersize, marker=style)

        if align == 'right':
            plt.xticks(
                [x + 0.5 for x in range(0, RANGE, 1)],
                range(MAXYEAR - RANGE, MAXYEAR, 1)
            )

        self.save_figure("Top 100 lifespans (%s)" % ", ".join([database_name, dtype] + fnargs))
        plt.show()

        print(", ".join("%d. %s" % (i, whats[seq[i]][0]) for i in range(len(whats))))
