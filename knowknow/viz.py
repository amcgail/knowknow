from . import plt, Path, np, Dataset, env

__all__ = [
    'yearly_counts','matrix'
]

def _normalize(mat, axis=0):
    if axis==0: # rows
        return mat/mat.sum(axis=1)[:,None]
    elif axis==1: #columns
         return (mat.T/mat.T.sum(axis=1)[:,None]).T
    else:
        raise Exception('axis != 0/1')

# colormaps...
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

def matrix(db, typ='docs', plot=False, norm=None, trans=False, includeComplement=False, **kwargs):

    assert(isinstance(db, Dataset))

    cnt = db.by( *list(kwargs) )

    # by1=range(35,50), by2=range(10,20)
    if len(list(kwargs)) != 2:
        raise Exception("matrix with dim != 2 not supported")

    if typ == 'docs':
        c = cnt.docs
    elif typ == 'cits':
        c = cnt.cits
    else:
        raise Exception("wtf... docs or cits bro")

    key1, key2 = sorted(kwargs)
    by1 = kwargs[key1]
    by2 = kwargs[key2]

    import numpy as np
    from collections import defaultdict

    def mk_array():
        for a1 in by1:
            my_series = np.array([c[(a1, a2)] for a2 in by2])

            if includeComplement:
                out = 0
                for (pp1, pp2), count in c.items():
                    # to account for a temporary bug
                    if pp1 < 0 or pp2 < 0:
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
            for (pp1, pp2),count in c.items():
                # to account for a temporary bug
                if pp1 < 0 or pp2 < 0:
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

    if norm is not None:
        from . import matrix_normalize
        if norm not in kwargs:
            raise Exception('norm should be the axis along which you want to normalize')

        norm_i = sorted(kwargs).index(norm)
        mat = matrix_normalize(mat, axis=norm_i)

    if trans:
        mat = mat.transpose()

    if plot:
        from . import plt
        fig = plt.figure(figsize=(20, 4))  # plt.subplots()#
        plt.imshow(mat, interpolation='nearest', cmap=plt.cm.hsv)
        # IMS.cmap.set_under('yellow')
        # IMS.cmap.set_over('orange')
        plt.colorbar()

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
    else:
        return mat

def yearly_counts(names, dataset, myname=None, 
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