from collections import Counter

import knowknow
import matplotlib.pyplot as plt
import numpy
import seaborn

plt.style.use('seaborn-whitegrid')


def badass_heatmap(whats, fnargs=[], RANGE=None,
                   markers={}, markersize=50, align='left',
                   proportional=False, MAXYEAR=2018,
                   **kwargs):
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

    knowknow.save_figure("Top 100 lifespans (%s)" % ", ".join([database_name, dtype] + fnargs))
    plt.show()

    print(", ".join("%d. %s" % (i, whats[seq[i]][0]) for i in range(len(whats))))


if __name__ == '__main__':
    dtype = 'c'
    # database_name = "sociology-jstor-basical"
    database_name = 'sociology-wos'
    basedir = '/home/ishsonya/workspace/knowknowlib/example'
    engine = knowknow.KnowKnow(BASEDIR=basedir, NB_DIR=basedir)
    ify = knowknow.comb(dtype, 'fy')  # TODO: what is ify?

    cnt = engine.get_cnt(name='%s.doc' % database_name, keys=['fy', ify, dtype])  # TODO
    ysum = engine.load_variable('%s.%s.ysum' % (database_name, dtype))  # TODO

    # TODO: understand why it's needed
    #  I think it loads some .fy and .ysum files. What is stored there is unclear
    """Loaded keys: dict_keys(['fy', 'fy.t', 't'])
    Available keys: ['c', 'c.c', 'c.c.fy', 'c.fa', 'c.fj', 'c.fy', 'c.t', 'fa', 'fa.fj.fy', 'fj', 'fj.fy', 'fj.t', 'fy', 'fy.t', 't', 't.t']
    list(ysum)[:5]= ['relationship', 'multiple', 'protestant', 'religious', 'thesis']
    Counter(dict(cnt[dtype])).most_common(3) = [(t(t='social'), 25329), (t(t='one'), 20959), (t(t='also'), 20860),]
    """

    all_years = numpy.array([[1, 2], [3, 4]])

    whats = [(x,) for x in ysum if (50 < ysum[x]['total'] < 1000)]
    engine.badass_heatmap(whats=whats,
                          fnargs=['random', 'raw'],
                          proportional='rows',
                          align='right',
                          RANGE=40,
                          MAXYEAR=2000,
                          cnt=cnt,
                          dtype=dtype,
                          ysum=ysum,
                          ify=ify,
                          database_name=database_name)

    whats = Counter(dict(cnt[dtype].items())).most_common(150)[50:100]
    whats = [x[0] for x in whats]
    engine.badass_heatmap(whats=whats,
                          fnargs=['most_cits', 'raw'],
                          align='right',
                          cnt=cnt,
                          dtype=dtype,
                          ysum=ysum,
                          ify=ify,
                          database_name=database_name)

    # aim: to sort by something else interesting.
    # I chose date of publication!!

    for decade in range(1950, 2020, 10):

        names = list(cnt[ify].keys())
        names = [getattr(x, dtype) for x in names]
        names = [x for x in names if x in ysum]

        whats = sorted(cnt[ify], key=lambda x: -ysum[getattr(x, dtype)]['total'] if getattr(x, dtype) in ysum else 0)
        whats = [x.ta for x in whats]
        whats = [x for x in whats if (x in ysum) and (decade <= ysum[x]['maxcounty'] < decade + 10)]
        print(len(whats), "total")

        whatskeep = set()
        i = 0
        while len(whatskeep) < 100 and i < len(whats):
            whatskeep.add(knowknow.make_cross(ta=whats[i]))
            i += 1
        whatskeep = list(whatskeep)

        cmap = seaborn.color_palette("cubehelix", 50)
        engine.badass_heatmap(
            whats=whatskeep,
            fnargs=['top_cit_%ss' % decade, 'raw'],
            RANGE=None,
            markers={x.ta: {decade + 10: "<"} for x in whatskeep},
            markersize=30,
            cmap=cmap,
            cnt=cnt,
            dtype=dtype,
            ysum=ysum,
            ify=ify,
            database_name=database_name
        )

        plt.show()
