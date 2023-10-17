def unique_legend(axis):
    import numpy as np
    handles, labels=axis.get_legend_handles_labels()
    labels, ind= np.unique(labels, return_index=True)
    return {'labels':list(labels), 'handles':list(np.array(handles)[ind])}
