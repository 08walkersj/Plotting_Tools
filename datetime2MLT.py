""" MLT UT Plot """
def UT_to_mlt(UT):
    from apexpy import Apex
    A= Apex()
    return [A.mlon2mlt(105, UT)]
import matplotlib as mpl
@mpl.ticker.FuncFormatter
def MLTformatter(x, pos):
    """Assuming input x is a matplotlib datenum, convert x to MLT"""
    return "{:04.1f}".format(UT_to_mlt(mpl.dates.num2date(x))[0])
