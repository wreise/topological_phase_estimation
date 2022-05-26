# Author(s): Wojciech Reise
#
# Copyright (C) 2021 Inria

import numpy as np

f_0 = lambda t: np.sin(2.*np.pi*t)
f_1 = lambda t:  2.0/1.76*(0.5*np.sin(2.*np.pi*(t-0.6)) + 0.5*np.sin(2.*2.*np.pi*(t-0.6)))
f_2 = lambda t: 2.0/1.91*(0.5*np.sin(2.*np.pi*t) + 0.7*np.sin(2*2.*np.pi*(t-0.3))*np.sin(2.*np.pi*(t-0.3))) 


def f_3(t):
    a = 1./10.
    integral = np.floor(t)
    frac = t - integral
    y = np.sin(2.*np.pi*(
          frac/(4.*a)*(frac<=a) + (frac>a)*(frac<=1-a)*((frac-a)/(2.*(1.-2.*a))+1./4.)
          + (frac>=1.-a)*((frac-(1-a))/(4.*a) + 3./4.)))
    return y


def f_4(t):
    integral = np.floor(t)
    frac = t - integral
    y = 2.*((frac >= 0.1)*(frac-0.1)*10.*(frac<0.2) + (frac>= 0.2)*(frac<0.8) + (frac>=0.8)*(0.9-frac)*10.*(frac<0.9))-1
    return y


fs_easy = [f_0, f_1, f_2, f_3, f_4]
fs_easy_dict = {k: f for k, f in enumerate(fs_easy)}


def get_f(index):
    return fs_easy_dict[index]
