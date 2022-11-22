# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:18:23 2022

@author: antona

Plots the first 10 seconds of a slected subject form its session 1, run 1

# pip install plotly==5.11.0

Rendering options in Plotly:
https://stackoverflow.com/questions/35315726/plotly-how-to-display-charts-in-spyder
"""

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, DemonsP300 
from moabb.paradigms import P300
import mne

from plotly import tools
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
import plotly.io as pio
#pio.renderers.default = 'jpg'
pio.renderers.default = 'browser'

def  PlotEpochs(ds):
    
    subject_i = 4
    # load first subject
    subject = ds.get_data([subject_i])
    # subject 1, session 0, run 0
    run1 = list((list((list(subject.values())[0]).values())[0]).values())[0]
    
    raw = run1
    
    picks = mne.pick_types(raw.info, eeg = True, exclude=[])
    start, stop = raw.time_as_index([0, 10]) #first 10 seconds
    
    n_channels = 16 #currently it must be specified manually
    data, times = raw[picks[:n_channels], start:stop]
    ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]
    
    step = 1. / n_channels
    kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)
    
    # create objects for layout and traces
    layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
    traces = [Scatter(x=times, y=data.T[:, 0])]
    
    # loop over the channels
    for ii in range(1, n_channels):
            kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
            layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
            traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))
    
    # add channel names using Annotations
    annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                          text=ch_name, font=Font(size=9), showarrow=False)
                              for ii, ch_name in enumerate(ch_names)])
    layout.update(annotations=annotations)
    
    # set the size of the figure and plot it
    layout.update(autosize=False, width=1000, height=600)
    fig = Figure(data=Data(traces), layout=layout)
    #py.iplot(fig, filename='shared xaxis')
    fig.show()

#not available: bi2013a(), Lee2019_ERP()
for d in [bi2014a()]:
#for d in [BNCI2014009()]:
    PlotEpochs(d)
    print("======================================================================================")