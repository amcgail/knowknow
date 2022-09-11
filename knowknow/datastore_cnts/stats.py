from .. import defaultdict, np, pd

__all__ = ['top_decade_stratified', 'births_deaths']

def top_decade_stratified(dataset, what, yRange=None, percentile=None, topN=None, maxP=None, debug=True):
    if percentile is None and topN is None:
        print("Assuming top 1%. Use percentile or topN attributes")
        percentile = 0.01

    if percentile is not None and topN is not None:
        raise Exception('one or the other... not both percentile or topN attributes')

    if yRange is None and False:
        yRange = (
            dataset['RELIABLE_DATA_STARTS_HERE'],
            dataset['RELIABLE_DATA_ENDS_HERE']
        )

    start, end = yRange

    all_tops = set()
    first_top = {}

    # ranges loop from 1940-1950 to 1980-1990, in 1-year increments

    for RANGE_START, RANGE_END in zip(
            range(start, end - 10 + 1),
            range(start + 10, end + 1)
    ):

        count_in_range = defaultdict(int)
        for (item, fy), count in dataset.items(what, 'fy'):
            if count == 0:
                continue
            if RANGE_END >= fy >= RANGE_START:
                count_in_range[item] += count

        counts = list(count_in_range.values())
        if not len(counts):
            if debug: print("Skipping %s" % RANGE_START)
            continue

        if topN is not None and maxP is not None:
            if topN / len(counts) > maxP:
                if debug: print("skipping decade, maxP not reached...")
                continue

        if percentile is not None:
            q99 = np.quantile(np.array(counts), 1 - percentile)
            top1 = {k for k in count_in_range if count_in_range[k] >= q99}
        elif topN is not None:
            top1 = {k for k in sorted(count_in_range, key=lambda x: -count_in_range[x])[:topN]}
        else:
            raise Exception('neverrrr')

        for t in top1:
            if t not in all_tops:
                first_top[t] = RANGE_START
        all_tops.update(top1)

        if percentile is not None:
            if debug: print("%s /%s in the top %0.1f%% in %s,%s (%s total accumulated)" % (
                len(top1),
                len(count_in_range),
                percentile * 100,
                RANGE_START, RANGE_END,
                len(all_tops)
            ))
        else:
            if debug: print("%s /%s  in %s,%s (%s total accumulated)" % (
                len(top1),
                len(count_in_range),
                RANGE_START, RANGE_END,
                len(all_tops)
            ))

    def getrow(name):
        ret = dataset.trend(what, name, yRange).dump()
        ret['first_added'] = first_top[name]
        return ret

    alldf = pd.DataFrame.from_records([
        getrow(name)
        for name in all_tops
    ])

    alldf.fillna(value=np.nan, inplace=True)

    print(alldf.shape)
    return alldf


def births_deaths(tt,  # the trend this should be calculated on
                  birth_cutoff=0.5,  # relative to the last life
                  death_cutoff=0.1,  # relative to this life so far
                  forward_view=10,  # years to look forward for comparison
                  backward_view=5  # minimum years to look back for comparison
                  ):
    cits = tt.cits()

    births = [tt.first]
    deaths = []

    last_life_avg = None

    for y_spl in range(tt.first + backward_view, tt.data_end + 1 - forward_view):
        lastb = births[-1]

        if len(births) > len(deaths):  # alive now
            if y_spl < births[-1] + backward_view:
                continue
            before = tt.sum_between(lastb, y_spl - 1) / (y_spl - lastb)
            after = tt.sum_between(y_spl, y_spl + forward_view - 1) / forward_view

            ratio = after / before
            # print(ratio)

            if ratio <= death_cutoff:  # died
                last_life_avg = before
                deaths.append(y_spl)

        else:  # dead now
            if y_spl < deaths[-1] + forward_view:  # need some time for a rebirth
                continue
            if tt.sum_between(y_spl, y_spl) == 0:  # don't be reborn in a year you get no citations
                continue

            after = tt.sum_between(y_spl, y_spl + forward_view - 1) / forward_view

            if after / last_life_avg > birth_cutoff:  # reborn
                births.append(y_spl)

    return (births, deaths)