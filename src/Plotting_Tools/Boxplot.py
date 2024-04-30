import numpy as np
def add_boxkey(axis, size=8, x_center=.175, scale= False, ypos= 9.5, width=.8, boxplotkwargs={}, y_offset= 0.1, sample_data=False, **text_kwargs):
    if not sample_data:
        np.random.seed(342314)
        if not scale:
            scale= .03*(x_center/.175)
        sample_data = (np.random.normal(.2, scale=scale, size=(100,))+\
                np.random.normal(.15, scale=scale, size=(100,)))*(x_center/.175)
        sample_data[0]-=.15*(x_center/.175)
    boxkwargs=dict(vert=False, showmeans = True, meanline = True, 
                                positions=[ypos], widths= width)
    boxkwargs.update(boxplotkwargs)
    txt_kwargs= {'size':size}
    txt_kwargs.update(text_kwargs)
    print(boxkwargs)
    sample_boxplot = axis.boxplot(sample_data, **boxkwargs)
    text= [
        axis.text(sample_boxplot['medians'][0].get_xdata()[0], sample_boxplot['medians'][0].get_ydata()[0],
                s='Median', ha='center', va='top', color=sample_boxplot['medians'][0].get_color(), **txt_kwargs),
        axis.text(sample_boxplot['medians'][0].get_xdata()[0], sample_boxplot['medians'][0].get_ydata()[0]-y_offset*3,
                s='Interquartile\nRange', ha='center', va='top', color=sample_boxplot['boxes'][0].get_color(), **txt_kwargs),
        axis.text(sample_boxplot['means'][0].get_xdata()[0], sample_boxplot['means'][0].get_ydata()[-1],
                s='Mean', ha='center', va='bottom', color=sample_boxplot['means'][0].get_color(), **txt_kwargs),
        axis.text(sample_boxplot['caps'][0].get_xdata()[0], sample_boxplot['caps'][0].get_ydata()[-1],
                s='Lower\nQuartile', ha='center', va='bottom', color=sample_boxplot['caps'][0].get_color(), **txt_kwargs),
        axis.text(sample_boxplot['caps'][1].get_xdata()[0], sample_boxplot['caps'][1].get_ydata()[-1],
                s='Upper\nQuartile', ha='center', va='bottom', color=sample_boxplot['caps'][1].get_color(), **txt_kwargs),
        axis.text(sample_boxplot['fliers'][0].get_xdata()[0], sample_boxplot['fliers'][0].get_ydata()[-1]+y_offset,
                s='Outliers', ha='center', va='bottom', color=sample_boxplot['fliers'][0].get_color(), **txt_kwargs)]
    return sample_boxplot, text
