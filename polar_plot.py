#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:52:32 2020
@author: Simon Walker

"""
#@author: zef014
# import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import inspect
class ArgumentError(Exception):
     pass
def get_default_args(func):
    import inspect
    """ return dictionary of default arguments to func """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
def store(*args, **kwargs):
    import inspect
    func= args[-1]
    argdict = get_default_args(func) # start by collecting defaults
    argnames = inspect.getfullargspec(func)[0]
    
    l= len(args)
    print(args)
    if l>1:
        named_args = dict(zip(argnames[1:l], args[1:]))
        argdict.update(named_args) # add positional arguments
    print(argdict)
    
    argdict.update(kwargs)
    return func(**argdict), argdict
def store_properties(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        ax= args[0]
        argdict = get_default_args(func) # start by collecting defaults
        argnames = inspect.getfullargspec(func)[0]
        l= len(args)
        if l>1:
            named_args = dict(zip(argnames[1:l], args[1:]))
            argdict.update(named_args) # add positional arguments
        argdict.update(**kwargs)
        q= func(ax, **argdict)
        q.kwargs= argdict
        q.kwargs.update({'name':func.__name__})
        return q
    return wrapper

def polar(ax, hemisphere='Northern', mode='mag'):
    if mode.lower()=='mag':
        return polar_mag(ax, hemisphere=hemisphere)
    elif mode.lower()=='geo':
        return polar_geo(ax, hemisphere=hemisphere)
    else:
        raise ArgumentError("Incorrect mode input must be either 'mag' or 'geo'")
class polar_mag():
    def __init__(self, ax, hemisphere='Northern'):
        # self= ax.copy()
        # blindcopy(self, ax)
        # for att in inspect.getmembers(ax):
        #     try:
        #         exec('self.'+att[0]+'=ax.'+att[0])
        #     except:
        #         pass
        self.__dict__=ax.__dict__.copy()
        self.get_figure=ax.get_figure
        self.get_position= ax.get_position
        self._set_position= ax._set_position
        self.set_anchor= ax.set_anchor
        self.set_mlt_zero_location= ax.set_theta_zero_location
        self.set_title=ax.set_title
        self.set_zorder= ax.set_zorder
        # self.__delattr__= ax.__delattr__
        # self.__dir__= ax.__dir__
        # self.__doc__= ax.__doc__
        # self.__eq__= ax.__eq__
        # self.__format__= ax.__format__
        # self.__ge__= ax.__ge__
        # self.__getattribute__= ax.__getattribute__
        # self.__gt__= ax.__gt__
        # self.__hash__= ax.__hash__
        # self.__init__= ax.__init__
        # self.__init_subclass__= ax.__init_subclass__
        # self.__le__= ax.__le__
        # self.__lt__= ax.__lt__
        # self.__module__= ax.__module__
        if hemisphere.lower()=='northern':
            self.Hadj= +1 # Hemispheric Adjuster
        elif hemisphere.lower()=='southern':
            self.Hadj= -1 # Hemispheric Adjuster
        else:
            raise ArgumentError(f"hemisphere argument incorrect. Must be 'northern' or southern. argument was set to be {hemisphere}")
        ticks=list(range(0, 100, 10))
        ax.set_rticks(ticks[:int(len(ax.get_yticklabels())/2)+2])
        ax.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21])
        labels= np.array((range(90, -10, -10)))*self.Hadj
        ax.set_yticklabels(labels[:len(ax.get_yticklabels())])
        ax.set_theta_zero_location('S')
        self.old_plot= ax.plot
        self.old_scatter= ax.scatter
        self.set_colatmax= ax.set_rmax
        self.set_colatmin= ax.set_rmin
        self.old_contourf= ax.contourf
        self.old_pcolormesh= ax.pcolormesh
        self.old_fillbetween= ax.fill_between
        self.in_axes= ax.in_axes
        self.ax= ax
        self.old_quiver= ax.quiver
        self.in_axes= ax.in_axes
        self.set_xlabel= self.ax.set_xlabel
        self.set_ylabel= self.ax.set_ylabel
        ax.format_coord= self.make_format()
        ax.autoscale(enable=False)
        self.legend= ax.legend
        self.xaxis_inverted()
    
        # cursor = Cursor(ax)
        # ax.get_figure().canvas.mpl_connect('motion_notify_event', functools.partial(cursor.mouse_move, self))
    def xaxis_inverted(self, *args):
    	print('xaxis not inverted')
    	return False
    def yaxis_inverted(self, *args):
    	print('yaxis not inverted')
    	return False
    def set_xlim(self, *args):
    	print(f'set xlim not supported \n {args}')
    	return args
    def set_ylim(self, *args):
    	print(f'set ylim not supported \n {args}')
    	return args
    def set_xlabel(self, *args):
    	print(f'set_xlabel not supported \n {args}')
    	return args
    def set_ylabel(self, *args):
    	print(f'set_ylabel not supported \n {args}')
    	return args
    def set_latticks(self, ticks, labels=False, **kwargs):
        if not labels:
            labels=ticks
        self.ax.set_rticks(90-ticks, labels=ticks, **kwargs)
        self.ax.set_yticklabels(labels, **kwargs)
    def set_mltrange(self, mltrange, **kwargs):
        print(mltrange)
        theta, _ =self.conv(mltrange, [80]*2)
        theta= np.rad2deg(theta)
        theta[theta<0]+=360
        theta= np.round(theta)
        print(min(theta), max(theta))
        self.ax.set_thetamin(min(theta), **kwargs)
        self.ax.set_thetamax(max(theta), **kwargs)
        mlt_labels=np.arange(0, 24, 3)
        mlt_labels=mlt_labels[(mlt_labels>=mltrange[0]) & (mlt_labels<=mltrange[-1])]
        theta2, _= self.conv(mlt_labels, [80]*len(mlt_labels))
        self.ax.set_xticks(theta2)
        self.ax.set_xticklabels(mlt_labels)
        self.ax.set_thetamin(min(theta), **kwargs)
        self.ax.set_thetamax(max(theta), **kwargs)
    def conv(self, mlt, mlat):
        mlt=np.array(mlt)
        mlat=np.array(mlat*self.Hadj)
        theta=mlt*np.pi/12
        x= -np.sin(theta-np.pi/2)
        y= np.sin(theta)
        θ = np.arctan2(y, x)
        return θ, 90-mlat
    def conv_inv(self, theta, r):
        mlat= 90-np.array([r])
        mlt= np.array([theta]) *(12/np.pi)
        mlt[mlt<-0]+=24
        return mlt, mlat
    def vec_theta(self, mlt):
        theta=mlt*np.pi/12
        x= np.sin(theta)
        y= np.sin(theta-np.pi/2)
        θ = np.arctan2(y, x)
        return θ
    def vec_conv(self, dr, dt, theta):
        return dr * np.cos(theta) - dt*np.sin(theta), dr*np.sin(theta) + dt*np.cos(theta)
    def plot(self, mlt, mlat, **kwargs):
        return self.old_plot(*self.conv(mlt, mlat), **kwargs)
    def scatter(self, mlt, mlat, **kwargs):
        return self.old_scatter(*self.conv(mlt, mlat), **kwargs)
    def set_rmin(self, mlat):
        return self.set_colatmax(90-mlat)
    def set_rmax(self, mlat):
        return self.set_colatmin(90-mlat)
    def contourf(self, mlt, mlat, z, **kwargs):
        return self.old_contourf(*self.conv(mlt, mlat), z, **kwargs)
    def pcolormesh(self, mlt, mlat, z, **kwargs):
        return self.old_pcolormesh(*self.conv(mlt, mlat), z, **kwargs)
    def fill_between(self, mlt, mlat, **kwargs):
        return self.old_fillbetween(*self.conv(mlt, mlat), **kwargs)
    @store_properties
    def quiver(self, mlt, mlat, East, North, **kwargs):
        v_theta= self.vec_theta(mlt)
        theta, r= self.conv(mlt, mlat)
        return self.old_quiver(theta, r, *self.vec_conv(-North, East, v_theta), **kwargs)
    def hist2d(self, mlt, mlat , bins, hist_kwargs={}, pc_kwargs={}):
        c, x, y=np.histogram2d(mlt, mlat, bins= bins, **hist_kwargs)
        if 'weights' in hist_kwargs:
            hist_kwargs['weights']=None
            c2, x2, y2= np.histogram2d(mlt, mlat, bins=bins, **hist_kwargs)
            c/=c2
        return x, y, c, self.pcolormesh(*np.meshgrid(np.append(x, x[0]), y), np.append(c, [c[0]], axis=0).T, **pc_kwargs)
    def imshow(self, image, mlt=None, mlat=None, image_format='xarray', inimage='image', **kwargs):
        if image_format=='xarray':
            mlt, mlat= image.mlt.values, image.mlat.values
        mlt_ind, mlat_ind= np.isfinite(mlt), np.isfinite(mlat)
        self.pcolormesh(mlt[mlt_ind&mlat_ind], mlat[mlt_ind&mlat_ind], image['image'].values[mlt_ind&mlat_ind], **kwargs)
    def make_format(current):
        # current and other are axes
        def format_coord(theta, r):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.conv_inv(theta,r)
            # convert back to data coords with respect to ax
            ax_coord= (float(i) for i in display_coord)
            # coords = [ax_coord, (x, y)]
            string= 'mlt={:.2f}, mlat={:.2f}'.format(*ax_coord) 
            # 'θ={:.3f} r={:.3f}'.format(theta, r)
            return (string)
            # return ('Left: {:<40}    Right: {:<}'
            #         .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
        return format_coord
    def get_projected_coastlines(self, datetime, height=0, **kwargs):
        import cartopy.io.shapereader as shpreader
        from apexpy import Apex
        """ generate coastlines in projected coordinates """

        if 'resolution' not in kwargs.keys():
            kwargs['resolution'] = '50m'
        if 'category' not in kwargs.keys():
            kwargs['category'] = 'physical'
        if 'name' not in kwargs.keys():
            kwargs['name'] = 'coastline'

        shpfilename = shpreader.natural_earth(**kwargs)
        reader = shpreader.Reader(shpfilename)
        coastlines = reader.records()
        multilinestrings = []
        A = Apex(date=datetime)
        for coastline in coastlines:
            if coastline.geometry.geom_type == 'MultiLineString':
                multilinestrings.append(coastline.geometry)
                continue
            lon, lat = np.array(coastline.geometry.coords[:]).T 
            mlat, mlon= A.geo2apex(lat, lon, height)
            mlt= A.mlon2mlt(mlon, datetime)
            yield mlt,mlat

        for mls in multilinestrings:
            for ls in mls:
                lon, lat = np.array(ls.coords[:]).T
                mlat, mlon= A.geo2apex(lat, lon, height)
                mlt= A.mlon2mlt(mlon, datetime)
                yield mlt, mlat
    def coastlines(self, datetime, height=0, map_kwargs=None, plot_kwargs=None):
        if (plot_kwargs is None):
            plot_kwargs= {'color':'k'}
        elif not('color' in plot_kwargs.keys()):
            plot_kwargs.update({'color':'k'})
        plots=[]
        if map_kwargs is None:
            for line in self.get_projected_coastlines(datetime,height=height):
                plots.extend(self.plot(line[0], line[1], **plot_kwargs))
        else:
            for line in self.get_projected_coastlines(datetime,height=height, **map_kwargs):
                plots.extend(self.plot(line[0], line[1], **plot_kwargs))
            
        return plots
class polar_geo():
    def __init__(self, ax, hemisphere='Northern'):
        # self= ax.copy()
        # blindcopy(self, ax)
        # for att in inspect.getmembers(ax):
        #     try:
        #         exec('self.'+att[0]+'=ax.'+att[0])
        #     except:
        #         pass
        self.__dict__=ax.__dict__.copy()
        self.get_figure=ax.get_figure
        self.get_position= ax.get_position
        self._set_position= ax._set_position
        self.set_anchor= ax.set_anchor
        self.set_mlt_zero_location= ax.set_theta_zero_location
        self.set_title=ax.set_title
        self.set_zorder= ax.set_zorder
        if hemisphere.lower()=='northern':
            self.Hadj= +1 # Hemispheric Adjuster
        elif hemisphere.lower()=='southern':
            self.Hadj= -1 # Hemispheric Adjuster
        else:
            raise ArgumentError(f"hemisphere argument incorrect. Must be 'northern' or southern. argument was set to be {hemisphere}")
        # self.__delattr__= ax.__delattr__
        # self.__dir__= ax.__dir__
        # self.__doc__= ax.__doc__
        # self.__eq__= ax.__eq__
        # self.__format__= ax.__format__
        # self.__ge__= ax.__ge__
        # self.__getattribute__= ax.__getattribute__
        # self.__gt__= ax.__gt__
        # self.__hash__= ax.__hash__
        # self.__init__= ax.__init__
        # self.__init_subclass__= ax.__init_subclass__
        # self.__le__= ax.__le__
        # self.__lt__= ax.__lt__
        # self.__module__= ax.__module__
        ticks=list(range(0, 100, 10))
        ax.set_rticks(ticks[:int(len(ax.get_yticklabels())/2)+2])
        ax.set_xticklabels(list(range(0, 360, 45)))
        labels= np.arange(90, -10, -10)*self.Hadj
        ax.set_yticklabels(labels[:len(ax.get_yticklabels())])
        ax.set_theta_zero_location('S')
        self.old_plot= ax.plot
        self.old_scatter= ax.scatter
        self.set_colatmax= ax.set_rmax
        self.set_colatmin= ax.set_rmin
        self.old_contourf= ax.contourf
        self.old_pcolormesh= ax.pcolormesh
        self.old_fillbetween= ax.fill_between
        self.in_axes= ax.in_axes
        self.ax= ax
        self.old_quiver= ax.quiver
        self.in_axes= ax.in_axes
        self.old_text= ax.text
        ax.format_coord= self.make_format()
        ax.autoscale(enable=False)
        self.legend= ax.legend

        # cursor = Cursor(ax)
        # ax.get_figure().canvas.mpl_connect('motion_notify_event', functools.partial(cursor.mouse_move, self))
    def set_xlim(self, *args):
    	print(f'set xlim not supported \n {args}')
    	return args
    def set_ylim(self, *args):
    	print(f'set ylim not supported \n {args}')
    	return args
    def set_xlabel(self, *args):
    	print(f'set_xlabel not supported \n {args}')
    	return args
    def set_rmin(self, mlat):
        return self.set_colatmax(90-mlat)
    def set_rmax(self, mlat):
        return self.set_colatmin(90-mlat)
    def set_ylabel(self, *args):
    	print(f'set_ylabel not supported \n {args}')
    	return args
    def xaxis_inverted(self, *args):
    	return args
    def yaxis_inverted(self, *args):
    	return args
    def set_latticks(self, ticks, labels=False, **kwargs):
        ticks*=self.Hadj
        if not labels:
            labels=ticks*self.Hadj
        self.ax.set_rticks(90-ticks, **kwargs)
        self.ax.set_yticklabels(labels, **kwargs)
    def set_mlonrange(self, mlonrange, **kwargs):
        print(mlonrange)
        theta, _ =self.conv(mlonrange, [80]*2)
        theta= np.rad2deg(theta)
        theta[theta<0]+=360
        theta= np.round(theta)
        print(min(theta), max(theta))
        self.ax.set_thetamin(min(theta), **kwargs)
        self.ax.set_thetamax(max(theta), **kwargs)
        mlon_labels=np.arange(0, 24, 3)
        mlon_labels=mlon_labels[(mlon_labels>=mlonrange[0]) & (mlon_labels<=mlonrange[-1])]
        theta2, _= self.conv(mlon_labels, [80]*len(mlon_labels))
        self.ax.set_xticks(theta2)
        self.ax.set_xticklabels(mlon_labels)
        self.ax.set_thetamin(min(theta), **kwargs)
        self.ax.set_thetamax(max(theta), **kwargs)
    def conv(self, mlon, mlat):
        mlon=np.array(mlon)
        mlat=np.array(mlat)*self.Hadj
        mlon[mlon<0]+=360
        mlon[mlon>360]-=360
        return np.deg2rad(mlon), 90-mlat
    def conv_inv(self, theta, r):
        mlat= 90-np.array([r])
        mlon= np.rad2deg(theta)
        return mlon, mlat*self.Hadj
    def vec_theta(self, mlon):
        theta=mlon*np.pi/12
        x= np.sin(theta)
        y= np.sin(theta-np.pi/2)
        θ = np.arctan2(y, x)
        return θ
    def vec_conv(self, dr, dt, theta):
        return dr * np.cos(theta) - dt*np.sin(theta), dr*np.sin(theta) + dt*np.cos(theta)
    def plot(self, mlon, mlat, **kwargs):
        return self.old_plot(*self.conv(mlon, mlat), **kwargs)
    def scatter(self, mlon, mlat, **kwargs):
        return self.old_scatter(*self.conv(mlon, mlat), **kwargs)
    def set_rmin(self, mlat):
        return self.set_colatmax(90-mlat)
    def set_rmax(self, mlat):
        return self.set_colatmin(90-mlat)
    def contourf(self, mlon, mlat, z, **kwargs):
        return self.old_contourf(*self.conv(mlon, mlat), z, **kwargs)
    def pcolormesh(self, mlon, mlat, z, **kwargs):
        return self.old_pcolormesh(*self.conv(mlon, mlat), z, **kwargs)
    def fill_between(self, mlon, mlat, **kwargs):
        return self.old_fillbetween(*self.conv(mlon, mlat), **kwargs)
    def quiver(self, mlon, mlat, East, North, **kwargs):
        v_theta= self.vec_theta(mlon)
        theta, r= self.conv(mlon, mlat)
        return self.old_quiver(theta, r, *self.vec_conv(-North, East, v_theta), **kwargs)
    def hist2d(self, mlon, mlat , bins, hist_kwargs={}, pc_kwargs={}):
        c, x, y=np.histogram2d(mlon, mlat, bins= bins, **hist_kwargs)
        if 'weights' in hist_kwargs:
            hist_kwargs['weights']=None
            c2, x2, y2= np.histogram2d(mlon, mlat, bins=bins, **hist_kwargs)
            c/=c2
        return x, y, c, self.pcolormesh(*np.meshgrid(np.append(x, x[0]), y), np.append(c, [c[0]], axis=0).T, **pc_kwargs)
    def make_format(current):
        # current and other are axes
        def format_coord(theta, r):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.conv_inv(theta,r)
            if display_coord[0]<=0 and display_coord[0]>=-270:
                display_coord= display_coord[0]+360, display_coord[1]
            # convert back to data coords with respect to ax
            ax_coord= (float(i) for i in display_coord)
            # coords = [ax_coord, (x, y)]
            string= 'lon={:.2f}, lat={:.2f}'.format(*ax_coord) 
            # 'θ={:.3f} r={:.3f}'.format(theta, r)
            return (string)
            # return ('Left: {:<40}    Right: {:<}'
            #         .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
        return format_coord
    def get_projected_coastlines(self, height=0, mag=False, **kwargs):
        import cartopy.io.shapereader as shpreader
        from apexpy import Apex
        """ generate coastlines in projected coordinates """

        if 'resolution' not in kwargs.keys():
            kwargs['resolution'] = '50m'
        if 'category' not in kwargs.keys():
            kwargs['category'] = 'physical'
        if 'name' not in kwargs.keys():
            kwargs['name'] = 'coastline'
        shpfilename = shpreader.natural_earth(**kwargs)
        reader = shpreader.Reader(shpfilename)
        coastlines = reader.records()
        multilinestrings = []
        for coastline in coastlines:
            if coastline.geometry.geom_type == 'MultiLineString':
                multilinestrings.append(coastline.geometry)
                continue
            lon, lat = np.array(coastline.geometry.coords[:]).T
            lat, lon= Apex().geo2qd(lat, lon, 0)
            yield lon,lat

        for mls in multilinestrings:
            for ls in mls:
                lon, lat = np.array(ls.coords[:]).T
                lat, lon= Apex().geo2qd(lat, lon, 0)
                yield lon, lat
    def coastlines(self, height=0, map_kwargs=None, plot_kwargs=None):
        if (plot_kwargs is None):
            plot_kwargs= {'color':'k'}
        elif not('color' in plot_kwargs.keys()):
            plot_kwargs.update({'color':'k'})
        plots=[]
        if map_kwargs is None:
            for line in self.get_projected_coastlines(height=height):
                plots.extend(self.plot(line[0], line[1], **plot_kwargs))
        else:
            for line in self.get_projected_coastlines(height=height, **map_kwargs):
                plots.extend(self.plot(line[0], line[1], **plot_kwargs))
            
        return plots
    def text(self, mlon, mlat, s, **kwargs):
        return self.old_text(*self.conv(mlon, mlat), s, **kwargs)
RE= 6.371E6
class spherical():
    def __init__(self, ax, projection='geo'):
        self.ax= ax
        self.projection= projection
        # ax.format_coord= self.make_format()
    def lon_lat_r_to_xyz(self, mlon, mlat, r):
        mlon, mlat= np.deg2rad(mlon), np.deg2rad(mlat)
        x= r*np.cos(mlat)*np.sin(mlon)
        y= r*np.cos(mlat)*np.cos(mlon)
        z= r*np.sin(mlat)
        return x, y, z
    def xyz_to_lon_lat_r(self, x, y, z):
        mlon= np.arctan2(x,y)
        mlat= np.arctan2(z*np.cos(mlon),y)
        r= z/np.sin(mlat)
        r= np.sqrt(x**2+y**2+z**2)
        return np.rad2deg(mlon), np.rad2deg(mlat), r
    def ENU2ijk(self, mlon, mlat, East, North, Up):
        mlon, mlat = np.deg2rad(mlon), np.deg2rad(mlat)
        i= East*np.cos(mlon) + North*np.sin(mlat)*np.sin(mlon) + Up*np.cos(mlat)*np.sin(mlon)
        j= -East*np.sin(mlon) -North*np.sin(mlat)*np.cos(mlon) +Up*np.cos(mlat)*np.cos(mlon)
        k= North*np.cos(mlat) + Up*np.sin(mlat)
        return i, j, k
    def plot(self, lon ,lat, r, **kwargs):
        return self.ax.plot(*self.lon_lat_r_to_xyz(lon, lat, r), **kwargs)
    def scatter(self, lon ,lat, r, **kwargs):
        return self.ax.scatter(*self.lon_lat_r_to_xyz(lon, lat, r), **kwargs)
    def quiver(self, lon, lat, r, East, North, Up, **kwargs):
        self.ax.quiver(*ax.lon_lat_r_to_xyz(lon, lat, r), *ax.ENU2ijk(lon, lat, East, North, Up), **kwargs)
    def get_projected_coastlines(self, datetime, height=0, **kwargs):
        import cartopy.io.shapereader as shpreader
        from apexpy import Apex
        """ generate coastlines in projected coordinates """

        if 'resolution' not in kwargs.keys():
            kwargs['resolution'] = '50m'
        if 'category' not in kwargs.keys():
            kwargs['category'] = 'physical'
        if 'name' not in kwargs.keys():
            kwargs['name'] = 'coastline'

        shpfilename = shpreader.natural_earth(**kwargs)
        reader = shpreader.Reader(shpfilename)
        coastlines = reader.records()
        multilinestrings = []
        A = Apex(date=datetime)
        for coastline in coastlines:
            if coastline.geometry.geom_type == 'MultiLineString':
                multilinestrings.append(coastline.geometry)
                continue
            lon, lat = np.array(coastline.geometry.coords[:]).T 
            mlat, mlon= A.geo2apex(lat, lon, height)
            yield mlon,mlat

        for mls in multilinestrings:
            for ls in mls:
                lon, lat = np.array(ls.coords[:]).T
                mlat, mlon= A.geo2apex(lat, lon, height)
                yield mlon, mlat
    def coastlines(self, datetime, height=0, map_kwargs=None, plot_kwargs=None):
        if (plot_kwargs is None):
            plot_kwargs= {'color':'k'}
        elif not('color' in plot_kwargs.keys()):
            plot_kwargs.update({'color':'k'})
        plots=[]
        if map_kwargs is None:
            for line in self.get_projected_coastlines(datetime,height=height):
                plots.extend(self.plot(line[0], line[1], [RE]*len(line[0]), **plot_kwargs))
        else:
            for line in self.get_projected_coastlines(datetime,height=height, **map_kwargs):
                plots.extend(self.plot(line[0], line[1], [RE]*len(line[0]), **plot_kwargs))
            
        return plots
    def make_format(current):
        # current and other are axes
        def format_coord(x, y):
            from mpl_toolkits import mplot3d
            if current.ax.M is None:
                return {}
            def _line2d_seg_dist(p1, p2, p0):
                """distance(s) from line defined by p1 - p2 to point(s) p0
            
                p0[0] = x(s)
                p0[1] = y(s)
            
                intersection point p = p1 + u*(p2-p1)
                and intersection point lies within segment if u is between 0 and 1
                """
            
                x21 = p2[0] - p1[0]
                y21 = p2[1] - p1[1]
                x01 = np.asarray(p0[0]) - p1[0]
                y01 = np.asarray(p0[1]) - p1[1]
            
                u = (x01*x21 + y01*y21) / (x21**2 + y21**2)
                u = np.clip(u, 0, 1)
                d = np.hypot(x01 - u*x21, y01 - u*y21)

                return d
            def line2d_seg_dist(p1, p2, p0):
                """distance(s) from line defined by p1 - p2 to point(s) p0
            
                p0[0] = x(s)
                p0[1] = y(s)
            
                intersection point p = p1 + u*(p2-p1)
                and intersection point lies within segment if u is between 0 and 1
                """
                return _line2d_seg_dist(p1, p2, p0)
            p = (x, y)
            edges = current.ax.tunit_edges()
            ldists = [(line2d_seg_dist(p0, p1, p), i) for \
                        i, (p0, p1) in enumerate(edges)]
            ldists.sort()
        
            # nearest edge
            edgei = ldists[0][1]
        
            p0, p1 = edges[edgei]
        
            # scale the z value to match
            x0, y0, z0 = p0
            x1, y1, z1 = p1
            d0 = np.hypot(x0-x, y0-y)
            d1 = np.hypot(x1-x, y1-y)
            dt = d0+d1
            z = d1/dt * z0 + d0/dt * z1
        
            x, y, z = mplot3d.proj3d.inv_transform(x, y, z, current.ax.M)
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.xyz_to_lon_lat_r(x, y ,z)
            # convert back to data coords with respect to ax
            ax_coord= (float(i) for i in display_coord)
            # coords = [ax_coord, (x, y)]
            string= 'mlon={:.2f}, mlat={:.2f}, R={:.2f}'.format(*ax_coord) 
            # 'θ={:.3f} r={:.3f}'.format(theta, r)
            return (string)
            # return ('Left: {:<40}    Right: {:<}'
            #         .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
        return format_coord
if __name__=='__main__':
    from mpl_toolkits.mplot3d import Axes3D
    #Example Polar Plot
    fig= plt.figure()
    ax= polar(fig.add_subplot(111, projection='polar'))
    ax.quiver(np.array([0]), np.array([60]), np.array([0]), np.array([1]))
    1/0
    from datetime import datetime as dt
    datetime= dt(2008, 10, 3, 5, 30, 0, 0)
    p=ax.coastlines(datetime)
    fig= plt.figure()
    ax= spherical(fig.add_subplot(111, projection='3d'))
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*RE
    y = np.sin(u)*np.sin(v)*RE
    z = np.cos(v)*RE
    ax.ax.plot_surface(x, y, z, rstride=1, cstride=1,
        color='darkgreen', edgecolor=None, alpha=0.5)
    ax.coastlines(datetime)
    import numpy as np

    lon = np.linspace(0,360, 10)
    lat = np.linspace(-89, 89, 10)
    lon, lat = np.meshgrid(lon, lat)
    r= np.zeros(lon.shape)
    r[:]=RE
    North= np.ones(lon.shape)
    East= np.zeros(lon.shape)
    North[:]=15
    Up= np.zeros(lon.shape)
    ax.quiver(lon, lat, r, East, North, Up, color= 'red', length=100000, arrow_length_ratio=0.8)
