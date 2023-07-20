from email.policy import default
from matplotlib import cm
from . import *

__all__ = [
    'yearly_counts_grid','matrix','key2name','yearly_counts_table',
    'author_pub_table'
]

def key2name(tname, truncate=None):
    tstr = tname.split("|")
    if len(tstr) == 3:
        tstr = "%s (%s)" % (tstr[0], tstr[1])
    else:
        tstr = "%s %s" % (tstr[0], tstr[1])
    tstr = tstr.title()

    if truncate is not None:
        if tstr[:truncate] != tstr:
            tstr = tstr[:truncate] + "..."

    return tstr

# normalize the matrix by columns
def _normalize(mat, axis=0):
    if axis==0: # rows
        return mat/mat.sum(axis=1)[:,None]
    elif axis==1: #columns
         return (mat.T/mat.T.sum(axis=1)[:,None]).T
    else:
        raise Exception('axis != 0/1')

# colormaps...
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

def matrix(db, 
    typ='docs', plot=False, result=None, 
    norm=None, trans=False, includeComplement=False, 
    plt_kwargs={}, remove_cohorts=None, filt=None, 
    vrange_cutoff=0.001,
    **kwargs
):
    
    from collections import OrderedDict

    assert(isinstance(db, Dataset) or isinstance(db, datastore_cnts.counter.CountHelper))

    if plot and result is None:
        result = False
    elif result is None:
        result = True

    if 'by' not in kwargs:

        # by1=range(35,50), by2=range(10,20)
        if len(list(kwargs)) != 2:
            raise Exception("matrix with dim != 2 not supported")

        keys = key1,key2 = sorted(kwargs)
        by1 = kwargs[key1]
        by2 = kwargs[key2]

    else:

        # thought about making ordering a thing in general,
        # but it'll break all my old code...

        by = kwargs['by']
        assert len(by) == 2
        if type(by) in {dict,OrderedDict}:
            keys = key1,key2 = list(by)
            by1,by2 = by.values()
        else:
            raise("could be something here")


    import numpy as np
    from collections import defaultdict

    def mk_array():
        for a1 in by1:
            my_series = np.array([
                db(**{key1:a1, key2:a2}) 
                if (filt is None or filt(a1,a2))
                else 0
                for a2 in by2 
            ])

            if includeComplement:
                out = 0
                for (pp1, pp2), count in db.items(key1, key2):
                    # to account for a temporary bug
                    if pp1 < 0 or pp2 < 0:
                        continue
                    if filt is not None and not filt(pp1,pp2):
                        continue

                    # skip what's already included
                    if pp1 != a1:
                        continue
                    if pp2 in by2:
                        continue

                    out += count

                my_series = np.array(list(my_series)+[out])

            yield my_series

        if includeComplement:
            cLast = defaultdict(int)

            outside_both = 0
            for (pp1, pp2),count in db.items(key1, key2):
                # to account for a temporary bug
                if pp1 < 0 or pp2 < 0:
                    continue
                if filt is not None and not filt(pp1,pp2):
                    continue

                # skip what's already included
                if pp1 in by1:
                    continue

                if pp2 not in by2:
                    outside_both += count

                #print(pp1, pp2, count)
                cLast[pp2] += count

            cLast = [cLast[x] for x in by2] + [outside_both]
            #print(cLast)
            yield cLast


    mat = np.array(list(mk_array()))
    #print(mat)

    if remove_cohorts is not None:
        assert(type(remove_cohorts) == dict)
        assert(len(list(remove_cohorts)) == 1)
        assert('fy' in keys)

        k,v = list(remove_cohorts.items())[0]
        assert k in keys
        if k == key1:
            for jj in v: # loop over cohorts
                # loop over years
                for y in by2:
                    if y < jj:
                        continue

                    if y-jj >= max(by1)-1:
                        continue

                    mat[y-jj,y-min(by2)] = 0 # remove em
        if k == key2:
            for jj in v: # loop over cohorts
                # loop over years
                for y in by1:
                    if y < jj:
                        continue

                    if y-jj >= max(by1)-1:
                        continue

                    mat[y-min(by1),y-jj] = 0 # remove em

    if norm is not None:
        from . import matrix_normalize
        if norm not in keys:
            raise Exception('norm should be the axis along which you want to normalize')

        norm_i = keys.index(norm)
        mat = matrix_normalize(mat, axis=norm_i)

    if trans:
        mat = mat.transpose()

    if plot:
        from . import plt
        fig = plt.figure(figsize=(20, 4))  # plt.subplots()#

        mn = min(vrange_cutoff, 1-vrange_cutoff)
        default_plt_args = {
            'vmin':max(0,np.quantile(mat, mn)),
            'vmax':min(1,np.quantile(mat, 1-mn)),
            'cmap': cm.jet
        }

        plt_kwargs = dict(default_plt_args, **plt_kwargs)

        plt.imshow(mat, interpolation='nearest', 
            **plt_kwargs
        )
        #plt.colorbar();

        if trans:
            xtick = by1
            ytick = by2
        else:
            xtick = by2
            ytick = by1

        tick_target = 5

        xstep = max(len(xtick) // tick_target,1)
        ystep = max(len(ytick) // tick_target,1)

        plt.xticks(
            range(0, len(xtick),xstep),
            xtick[::xstep],
        )
        plt.yticks(
            range(0, len(ytick),ystep),
            ytick[::ystep],
        )

        if trans:
            plt.xlabel(key1)
            plt.ylabel(key2)
        else:
            plt.xlabel(key2)
            plt.ylabel(key1)
    
    if result:
        return mat

def yearly_counts_grid(names, dataset, myname=None,
    overwrite=True, 
    markers={}, print_names=None, 
    ctype='c', yearly_prop=False, 
    xlim=None, cols = 5,
    rows_per_group = 2,
        count_unit='doc', label_num='max'):

    assert(type(dataset) == Dataset)

    if xlim is None:
        xlim = (dataset.start, dataset.end)

    eps = (xlim[1]-xlim[0])/10
    xlim_display = (xlim[0]-eps, xlim[1]+eps)

    doing_both = False
    if type(count_unit) != str:
        if set(count_unit) == {'doc','cit'}:
            doing_both = True
        else:
            raise Exception('whoopsie')

    rows = len(names) // cols + bool(len(names) % cols) # = 15 for 5
    
    groupsize = rows_per_group * cols
    gs = rows // rows_per_group + bool(rows % rows_per_group)

    existing_files = Path(env.figure_dir).glob("%s.*" % myname)
    for fn in existing_files:
        fn.unlink()

    for groupi in range(gs):
        myFnName = "%s.%s" % (myname,groupi)
        if Path(myFnName).exists() and not overwrite:
            continue
        
        plt.figure(figsize=(cols*4,rows*1.2))
        plt.subplots_adjust(hspace=0.6, wspace=0)

        for i,name in enumerate(names[ groupi*groupsize : (groupi+1)*groupsize]):
            print(name)
            
            plt.subplot(rows,cols,i+1)

            yrstart,yrend = xlim
            years = range(yrstart, yrend+1)
            
            if count_unit == 'cit' or 'cit' in count_unit:
                vals = dataset.trend(ctype, name).cits(yrstart, yrend)
                
                if yearly_prop:
                    all_by_year = [ dataset(fy=YY).cits for YY in years ]
                    vals = np.array(vals) / np.array(all_by_year)

            else:
                vals = dataset.trend(ctype, name).docs(yrstart,yrend)
                
                if yearly_prop:
                    all_by_year = [ dataset(fy=YY).docs for YY in years ]
                    vals = np.array(vals) / np.array(all_by_year)

            # fill that shit up
            plt.fill_between(years,vals,color='black',alpha=0.4)


            if doing_both:
                vals2 = np.array(dataset.trend(ctype, name).docs())
                
                if yearly_prop:
                    all_by_year = [ dataset(fy=YY).docs for YY in years ]
                    vals2 = np.array(vals2) / np.array(all_by_year)
                    
                # this one is scaled...
                vals2 = np.max(vals) * vals2 * 0.5 / np.max(vals2)

                plt.plot(years, vals2, color='black')
            
            if print_names is not None:
                title = print_names(name)
            else:
                title = name
                
            if len(title) >= 40:
                title = title[:37]+"..."
                
            #t = plt.text(min(years) + (max(years)-min(years))/2, 1.2*(max(vals))/1,title, fontsize=13)
            plt.title(title)
            
            plt.axis('off')

            from math import ceil, floor

            XAXIS_YSHIFT = -max(vals)/25
            lines = []

            if False:
                # lines on the x-axis...
                for decade in range(floor(yrstart/10)*10,ceil(yrend/10)*10,10):
                    lines += [
                        (decade+1, decade+10-1), 
                        #(-max(vals)/5, -max(vals)/5), 
                        #(0,0),
                        (XAXIS_YSHIFT, XAXIS_YSHIFT), 
                        'black'
                    ]

            # labeling the last
            if label_num == 'last':
                last_num = vals[-1]

                if yearly_prop:
                    last_nump = "%0.1f%%"%(last_num*100)
                else:
                    last_nump = last_num

                lines += [
                    (xlim_display[1]-10, xlim_display[1]-5),
                    (last_num, last_num),
                    "black"
                ]
                plt.text(xlim_display[1]-3, last_num, last_nump, fontsize=12, verticalalignment='center', horizontalalignment='left')#*3+min(vals)
                
            elif label_num == 'max':
                #print(vals, title)
                max_num = np.max(vals)

                if yearly_prop:
                    max_nump = "%0.1f%%"%(max_num*100)
                else:
                    max_nump = max_num

                lines += [
                    (xlim_display[1]-10, xlim_display[1]-5),
                    (max_num, max_num),
                    "black"
                ]
                plt.text(xlim_display[1]-3, max_num, max_nump, fontsize=12, verticalalignment='center', horizontalalignment='left')#*3+min(vals)

            elif label_num is not None:
                raise Exception('don\'t recognize that label_num...')

            plt.plot(*lines)

            # dots on the x-axis
            half_centuries = range( ceil(xlim[0]/50)*50, floor(xlim[1]/50)*50 + 50, 50 )
            decades = list(set(range( ceil(xlim[0]/10)*10, floor(xlim[1]/10)*10 + 10, 10 )).difference(set(half_centuries)))
            plt.scatter(
                decades,
                [0]*len(decades),
                color='black',
                s=5
            )

            #print('max',np.max(vals))
            #print(list(vals))
            if True:
                plt.vlines(
                    half_centuries,
                    -np.max(vals)/5,
                    0
                )
            
            plt.xlim(xlim_display)
            plt.ylim((-np.max(vals)/5,np.max(vals)))
            
            # put the markers on the time seriesss
            if name in markers:
                ms = markers[ name ]
                if type(ms) == list:
                    #print('here!')
                    plt.scatter(
                        ms,
                        [-max(vals)/10]*len(ms),
                        color='red',
                        s=30,
                        marker="^"
                    )
                elif type(ms) == dict:
                    for k,v in ms.items():
                        plt.text(k, -max(vals)/2, v)

        dataset.save_figure(myFnName)
        plt.show();



def yearly_counts_table_simp(
        dta, who,
        NCOLS=2, NPAGES=None,
        markranges={}, yearlim=(1950,2015),
        tickstep=20,
        print_names={}, number_start=1,
        print_numbers = False):

    wper = 16 / 3
    hper = 14 / 20
    NROWS = (len(who) // NCOLS) + int(len(who) % NCOLS != 0)

    # top_end = (2080-2015) /  (2015-1950)
    # bottom_end = (1950 - 1880) / (2015 - 1950)
    # year_wid = yearlim[1] - yearlim[0]
    yearmin, yearmax = yearlim

    # xmin, xmax = (yearlim[0] - bottom_end * year_wid, yearlim[1] + top_end * year_wid)
    # def dist_trans(nyears):
    #    return year_wid * nyears / (2015-1950)

    plt.figure(figsize=(wper * NCOLS, hper * NROWS))

    for col_i in range(NCOLS):

        plt.subplot(1, NCOLS, col_i + 1)

        trends = []
        for i, tname in enumerate(who[col_i * NROWS: (col_i + 1) * NROWS]):
            tt = np.array(list(dta.trend('c', tname, (yearmin, yearmax))))

            # print(yearmin,yearmax)
            trends.append((tname, tt))

        for i, (tname, tt) in enumerate(trends):
            ttm = tt.max()
            ttl = tt[-1]
            ttl15 = tt[-15:].mean()

            BASE = (NROWS - 1 - i) * 2

            tt = tt / ttm
            tt = tt + BASE

            years = range(yearmin, yearmax + 1)

            plt.axis('off')
            plt.fill_between(years, tt, y2=BASE, color='gray')

            if tname in print_names:
                tstr = print_names[tname]
            else:
                tstr = key2name(tname, truncate=25)

            tstr = "(%s) " % ((col_i) * NROWS + i + number_start) * print_numbers + tstr
            plt.text(2020, BASE, tstr, fontsize=13)
            # plt.text(yearmax + dist_trans(2030-2015), BASE, int(ttm), horizontalalignment='center')
            # plt.text(yearmax + dist_trans(2050-2015), BASE, int(ttl), horizontalalignment='center')
            # plt.text(yearmax + dist_trans(2070-2015), BASE, "%0.1f" % ttl15, horizontalalignment='center')
            # plt.text(2025, i*2, int(ttm), horizontalalignment='center')

            if tname in markranges:
                # xmin,xmax = plt.xlim()
                ymin, ymax = [BASE, BASE + 1.4]
                mk = markranges[tname]
                plt.fill_between(mk, [ymin, ymin], [ymax, ymax], color='blue', alpha=0.2)

        plt.vlines(range(yearmin, yearmax, tickstep), 0, NROWS * 2, color='black', alpha=0.3)

        # table lines...
        # plt.vlines([ yearmax + dist_trans(DD-2015) for DD in range(2020, 2080 + 20, 20)], 0, NROWS * 2, color='black', alpha=1)  # table lines

        for y in range(yearmin, yearmax, tickstep):
            plt.text(y, -1, y, horizontalalignment='center')

        plt.xlim(yearmin, yearmax + 50)


def yearly_counts_table(
        dta, who,
        NCOLS=2, NPAGES=None,
        markranges={}, yearlim=(1950,2015),
        tickstep=20,
        print_names={}, number_start=1
):
    if NPAGES is not None:
        nper = len(who) // NPAGES + int( len(who) % NPAGES != 0 )
        for i in range(NPAGES):
            yearly_counts_table(
                dta,
                who[i*nper:(i+1)*nper],
                NCOLS,None,
                markranges,
                yearlim,
                tickstep,
                print_names=print_names, number_start=number_start
            )
        return

    #print(yearlim)

    wper = 17 / 3
    hper = 10 / 20
    NROWS = (len(who) // NCOLS) + int(len(who) % NCOLS != 0)

    top_end = (2080-2015) /  (2015-1950)
    bottom_end = (1950 - 1850) / (2015 - 1950)
    year_wid = yearlim[1] - yearlim[0]
    yearmin, yearmax = yearlim

    xmin, xmax = (yearlim[0] - bottom_end * year_wid, yearlim[1] + top_end * year_wid)
    def dist_trans(nyears):
        return year_wid * nyears / (2015-1950)


    plt.figure(figsize=(wper * NCOLS, hper * NROWS))

    for col_i in range(NCOLS):

        plt.subplot(1, NCOLS, col_i + 1)

        trends = []
        for i, tname in enumerate(who[col_i * NROWS: (col_i + 1) * NROWS]):
            tt = np.array(list(dta.trend('c', tname, (yearmin, yearmax))))

            #print(yearmin,yearmax)
            trends.append((tname, tt))

        for i, (tname, tt) in enumerate(trends):
            ttm = tt.max()
            ttl = tt[-1]
            ttl15 = tt[-15:].mean()

            BASE = (NROWS - 1 - i) * 2

            tt = tt / ttm
            tt = tt + BASE

            years = range(yearmin, yearmax+1)

            plt.axis('off')
            plt.fill_between(years, tt, y2=BASE, color='gray')

            if tname in print_names:
                tstr = print_names[tname]
            else:
                tstr = key2name(tname, truncate=25)

            tstr = "(%s) %s" % ((col_i) * NROWS + i + number_start, tstr)
            plt.text(yearmin - dist_trans(1950-1820), BASE, tstr)
            plt.text(yearmax + dist_trans(2030-2015), BASE, int(ttm), horizontalalignment='center')
            plt.text(yearmax + dist_trans(2050-2015), BASE, int(ttl), horizontalalignment='center')
            plt.text(yearmax + dist_trans(2070-2015), BASE, "%0.1f" % ttl15, horizontalalignment='center')
            # plt.text(2025, i*2, int(ttm), horizontalalignment='center')

            if tname in markranges:
                #xmin,xmax = plt.xlim()
                ymin,ymax = [BASE,BASE+1.4]
                mk = markranges[tname]
                plt.fill_between( mk, [ymin,ymin], [ymax,ymax], color='blue', alpha=0.2 )

        plt.vlines(range(yearmin, yearmax, tickstep), 0, NROWS * 2, color='black', alpha=0.3)

        # table lines...
        plt.vlines([ yearmax + dist_trans(DD-2015) for DD in range(2020, 2080 + 20, 20)], 0, NROWS * 2, color='black', alpha=1)  # table lines

        for y in range(yearmin, yearmax, tickstep):
            plt.text(y, -1, y, horizontalalignment='center')

        plt.xlim(xmin, xmax)

    # plt.savefig("top_cited.png", dpi=300)

def author_pub_table(dta, who, **kwargs):
    yearly_counts_table(dta, dta.search('c', who + "|"), **kwargs)