# -*- coding: utf-8 -*-
"""
Created on Fri Aug 09 20:52:47 2013

@author: Arthur
"""

import math as m
import random as r
import matplotlib.pyplot as p
import scipy.fftpack as sfft
import scipy as sp
import pylab as plb
import numpy as np
from PyQt4 import QtGui
from PyQt4 import QtCore

class ScrollingToolQT(object):
    def __init__(self, fig):
        # Setup data range variables for scrolling
        self.fig = fig
        self.xmin, self.xmax = fig.axes[0].get_xlim()
        self.step = 1e-6 # axis units

        self.scale = 1 # conversion betweeen scrolling units and axis units

        # Retrive the QMainWindow used by current figure and add a toolbar
        # to host the new widgets
        QMainWin = fig.canvas.parent()
        toolbar = QtGui.QToolBar(QMainWin)
        QMainWin.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        # Create the slider and spinbox for x-axis scrolling in toolbar
        self.set_slider(toolbar)
        self.set_spinbox(toolbar)

        # Set the initial xlimits coherently with values in slider and spinbox
        self.set_xlim = self.fig.axes[0].set_xlim
        self.draw_idle = self.fig.canvas.draw_idle
        self.ax = self.fig.axes[0]
        self.set_xlim(0, self.step)
        self.fig.canvas.draw()

    def set_slider(self, parent):
        # Slider only support integer ranges so use ms as base unit
        smin, smax = self.xmin*self.scale, self.xmax*self.scale

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent=parent)
        self.slider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slider.setTickInterval((smax-smin)/10.)
        self.slider.setMinimum(smin)
        self.slider.setMaximum(smax-self.step*self.scale)
        self.slider.setSingleStep(self.step*self.scale/5.)
        self.slider.setPageStep(self.step*self.scale)
        self.slider.setValue(0)  # set the initial position
        self.slider.valueChanged.connect(self.xpos_changed)
        parent.addWidget(self.slider)

    def set_spinbox(self, parent):
        self.spinb = QtGui.QDoubleSpinBox(parent=parent)
        self.spinb.setDecimals(6)
        self.spinb.setRange(0,1)
        self.spinb.setSingleStep(1e-4)
        self.spinb.setSuffix(" s")
        self.spinb.setValue(self.step)   # set the initial width
        self.spinb.valueChanged.connect(self.xwidth_changed)
        parent.addWidget(self.spinb)

    def xpos_changed(self, pos):
        #pprint("Position (in scroll units) %f\n" %pos)
        #        self.pos = pos/self.scale
        pos /= self.scale
        self.set_xlim(pos, pos + self.step)
        self.draw_idle()

    def xwidth_changed(self, xwidth):
        #pprint("Width (axis units) %f\n" % step)
        if xwidth <= 0: return
        self.step = xwidth
        self.slider.setSingleStep(self.step*self.scale/5.)
        self.slider.setPageStep(self.step*self.scale)
        old_xlim = self.ax.get_xlim()
        self.xpos_changed(old_xlim[0] * self.scale)

def getFFT(sig,samp_per):
    """ Method to get FFT of signal. Passed in the signal and its time-scale/axis.
        Returns its magnitude spectrum and its frequency axis, in opposite order. """
    FFT = abs(sp.fft(sig))
    freqs = sfft.fftfreq(len(sig),samp_per)
    return freqs, FFT

def timeScale(TIME_INC, NUM_POINTS):
    # time variable in increments defined by TIME_INC
    t = [x * TIME_INC for x in range(0, NUM_POINTS)]
    return t

def genData(TIME_INC, is_noisy, t):
    # Mean and std deviation of gaussian distribution for:
    # Amplitude noise
    AMP_MEAN = 0
    AMP_STDDEV = 0.125
    # Phase noise distribution
    PHASE_MEAN = 0.8  #multiplied by random[0,1], so any value will scale the random mean
    PHASE_STDDEV = 0.1

    samp_freq = 1/TIME_INC

    # frequency of sine waves
    omega1 = long(3e4*2*np.pi)
    omega2 = long(2e4*2*np.pi)
    omega3 = long(1e4*2*np.pi)

    print"Freq: %.3e" % omega1,
    print ",  %.3e" % omega2,
    print ",  %.3e" % omega3
    print "Samp Freq:  %.3e" % samp_freq
    if is_noisy:
        # empty lists for noisy sine waves
        s1 = []
        s2 = []
        s3 = []
        # generate the noisy sine waves with AWGN and AWG-Phase-Noise
        for s in t:
            s1.append(r.gauss(AMP_MEAN,AMP_STDDEV)+m.sin(omega1*s+r.gauss(PHASE_MEAN*(r.random()-0.5), PHASE_STDDEV)))
        for s in t:
            s2.append(r.gauss(AMP_MEAN,AMP_STDDEV)+m.sin(omega2*s+r.gauss(PHASE_MEAN*(r.random()-0.5), PHASE_STDDEV)))
        for s in t:
            s3.append(r.gauss(AMP_MEAN,AMP_STDDEV)+m.sin(omega3*s+r.gauss(PHASE_MEAN*(r.random()-0.5), PHASE_STDDEV)))

    else:
        # empty lists for the clean sine waves
        s1 = []
        s2 = []
        s3 = []
        # generate the clean sine waves
        for s in t:
            s1.append(m.sin(omega1*s))
        for s in t:
            s2.append(m.sin(omega2*s))
        for s in t:
            s3.append(m.sin(omega3*s))
        # add the waves together
    s_sum = [x1+x2+x3 for x1,x2,x3 in zip(s1,s2,s3)]

    return s_sum

def testPlot():
    """ Get/generate the data to play with """
    TIME_INC = 1e-6
    NUM_POINTS = 40000
    t = timeScale(TIME_INC,NUM_POINTS)
    noisy_sig = genData(TIME_INC,True, t)
    clean_sig = genData(TIME_INC,False,t)
    """ Get FFT of signal and the sampling frequency from the time intervals used to generate the signals"""
    freq, s_fft  = getFFT(noisy_sig, TIME_INC)
    freq2,s_fft2 = getFFT(clean_sig, TIME_INC)


    """ Show in 2 subplots the signals and their spectrums"""
    plb.subplot(211,axisbg='#FFFFCC')
    p.plot(t,clean_sig,'b')
    p.hold(True)
    p.grid(True)
    p.plot(t,noisy_sig,'r')
    plb.subplot(212,axisbg='#FFFFCC')
    #p.hold(False)
    p.plot(freq2, 20*sp.log10(s_fft2),'x-b')
    p.hold(True)
    p.plot(freq,  20*sp.log10(s_fft), '+-r')
    p.xticks([-10e4,-5e4,-4e4,-3e4,-2e4,-1e4,0,1e4,2e4,3e4,4e4,5e4,10e4])
    p.xlim([-1e5,1e5])
    p.grid(True)
    #p.show()
    q = ScrollingToolQT(p.gcf())
    return q   # WARNING: it's important to return this object otherwise
                   # python will delete the reference and the GUI will not respond!
## Find all points greater than N
#N = 5
#j2 = [i for i in j if i >= N]

if __name__ == "__main__":
    qpl = testPlot()
    p.show()


