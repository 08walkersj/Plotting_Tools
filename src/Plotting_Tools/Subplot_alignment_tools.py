import inspect
import numpy as np
class ArgumentError(Exception):
     pass
def subplot_matchx(axis1, axis2):
    """

    Parameters
    ----------
    axis1 : matplotlib subplot
        subplot to be adjusted.
    axis2 : matplotlib subplot
        subplots who's x dimensions is to be matched.

    Returns
    -------
    Set position output.

    """
    def onresize(axis1, axis2, event):
        pos= axis2.get_position()
        return axis1.set_position([pos.x0, axis1.get_position().y0, pos.width, axis1.get_position().height])
    import functools
    onresize_wrapper=functools.partial(onresize, axis1, axis2)
    cid = axis1.figure.canvas.mpl_connect('resize_event', onresize_wrapper)
    return onresize(axis1, axis2, None)
def subplot_matchy(axis1, axis2):
    """

    Parameters
    ----------
    axis1 : matplotlib subplot
        subplot to be adjusted.
    axis2 : matplotlib subplot
        subplots who's y dimensions is to be matched.

    Returns
    -------
    Set position output.

    """
    def onresize(axis1, axis2, event):
        pos= axis2.get_position()
        return axis1.set_position([axis1.get_position().x0, pos.y0, axis1.get_position().width, pos.height])
    import functools
    onresize_wrapper=functools.partial(onresize, axis1, axis2)
    cid = axis1.figure.canvas.mpl_connect('resize_event', onresize_wrapper)
    return onresize(axis1, axis2, None)

def subplot_align(axis1, *axes, dim='x'):
    if not dim.lower() in ['x', 'y', 'both']:
        raise ArgumentError(f'dimension to align is not understood. You chose: {dim}. Please specify either "x" or "y" or "both".')
    def onresize(axis1, axes, event):
        if dim.lower()=='x':
            x=np.concatenate([[ax.get_position().x0, ax.get_position().x1] for ax in axes])
            width= max(x)-min(x)
            x= min(x)
            y= axis1.get_position().y0
            height= axis1.get_position().height
        elif dim.lower()=='y':
            y=np.concatenate([[ax.get_position().y0, ax.get_position().y1] for ax in axes])
            height= max(y)-min(y)
            y= min(y)
            x= axis1.get_position().x0
            width= axis1.get_position().width
        elif dim.lower()=='both':
            x=np.concatenate([[ax.get_position().x0, ax.get_position().x1] for ax in axes])
            width= max(x)-min(x)
            x= min(x)
            y=np.concatenate([[ax.get_position().y0, ax.get_position().y1] for ax in axes])
            height= max(y)-min(y)
            y= min(y)
        return axis1.set_position([x, y, width, height])
    import functools
    onresize_wrapper=functools.partial(onresize, axis1, axes)
    cid = axis1.figure.canvas.mpl_connect('resize_event', onresize_wrapper)
    return onresize(axis1, axes, None)
