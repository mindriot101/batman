#!/usr/bin/env python
# -*- coding: utf-8 -*-

import batman
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


hjd = np.loadtxt('testtimes.txt')

params = batman.TransitParams()
params.t0 = 0.
params.per = 1.
params.rp = 0.1
params.a = 15.
params.inc = 87.
params.ecc = 0.
params.w = 90.
params.u = [
	0.7692,
	-0.716,
	1.1874,
	-0.5372,
]
params.limb_dark = 'nonlinear'

model = batman.TransitModel(params, hjd)
flux = model.light_curve(params)


c_lightcurve = np.loadtxt('c-lightcurve.txt', unpack=True)

print('Lightcurves match: %s' % np.allclose(flux, c_lightcurve[1]))

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
axes[0].plot(hjd, flux, 'k.')
axes[1].plot(c_lightcurve[0], c_lightcurve[1], 'k.')

axes[0].set(title='Python')
axes[1].set(title='C')
fig.savefig('comparison.png')
plt.close(fig)
