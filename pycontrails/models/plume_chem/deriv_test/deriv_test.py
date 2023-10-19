import numpy as np
import deriv_test

print(deriv_test.deriv.__doc__)

rc = np.ones([2, 2, 2, 510])
dj = np.ones([2, 2, 2, 96])
fl = np.ones([2, 2, 2, 606])
em = np.ones([2, 2, 2, 220])
h2o = np.ones([2, 2, 2])
m = np.ones([2])
o2 = np.ones([2, 2, 2])
yp = np.ones([2, 2, 2, 220])
y = np.ones([2, 2, 2, 220])
dts = 10

fl,y,ro2 = deriv_test.deriv(rc,fl,dj,em,h2o,m,o2,yp,y,dts)

print(y[:, :, :, 0])
