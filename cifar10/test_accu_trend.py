import numpy as np

class ExponentialMovingAverage(object):
    def __init__(self, decay=0.9):
        self.data = []
        self.decay = decay
        self.avg_val = 0.0

    def push(self, current_data):
        self.avg_val = self.decay * self.avg_val + (1 - self.decay) * current_data
        self.data.append(self.avg_val)

    def get(self):
        return self.data


accu=np.array([54.950, 71.720, 74.470, 78.800, 76.560, 80.330, 82.150, 81.020, 83.260, 85.580, 84.090, 82.760, 84.610, 86.980, 86.650, 86.510, 86.790, 87.630, 87.690, 87.750, 89.120, 88.060, 87.980, 89.090, 88.600, 88.560, 89.310, 90.090, 88.670, 89.570, 90.150, 89.260, 89.470, 89.710, 90.720, 90.480, 89.520, 88.440, 90.650, 90.600, 90.410, 90.670, 89.070, 90.780, 90.630, 90.100, 90.620, 90.130, 90.330, 91.220, 90.530, 90.830, 90.550, 91.110, 90.380, 89.970, 90.390, 90.050, 90.100, 90.520, 91.680, 91.120, 90.760, 90.400, 91.080, 91.320, 90.330, 90.990, 91.210, 90.780, 90.560, 90.980, 90.980, 91.080, 91.570, 90.900, 90.820, 90.850, 90.650, 91.280, 91.300, 91.690, 91.640, 90.150, 90.790, 91.180, 91.210, 91.420, 91.120, 90.930, 91.180, 91.210, 90.700, 91.510, 90.880, 91.470, 91.100, 91.170, 90.530, 91.190, 91.270, 91.840, 91.530, 91.400, 91.640, 91.640, 91.450, 91.770, 91.220, 90.920, 91.690, 91.040, 91.600, 91.410, 91.610, 91.730, 91.080, 91.160, 91.220, 91.790, 90.200, 91.410, 91.360, 91.460, 91.660, 91.290, 91.060, 92.040, 90.980, 90.930, 91.600, 91.820, 91.040, 91.090, 91.460, 91.660, 91.830, 91.240, 91.270, 91.470, 91.730, 91.060, 91.050, 91.510, 91.820, 91.180, 91.320, 91.310, 91.870, 91.420, 91.640, 91.320, 91.710, 91.660, 91.040, 91.330, 91.590, 91.440, 91.620, 91.220, 91.670, 91.400, 91.160, 92.080, 91.530, 91.630, 91.550, 91.770, 91.000, 92.220, 91.770, 91.200, 91.820, 91.570, 91.770, 91.890, 91.590, 90.920, 92.060, 91.430, 91.800, 91.450, 91.310, 92.090, 91.350, 91.870, 91.660, 91.440, 92.100, 92.140, 91.670, 91.670, 91.310, 91.930, 91.800, 91.710, 91.520, 92.000, 90.990, 91.520, 91.320, 91.610, 91.190, 91.600, 91.640, 91.680, 91.770, 91.620, 91.420, 91.240, 90.980, 91.760, 91.460, 91.530, 91.460, 92.110, 91.110, 91.400, 91.000, 91.100, 92.090, 91.510, 91.980, 91.720, 91.410, 91.660, 91.470, 91.840, 91.580, 91.840, 91.920, 91.020, 91.640, 91.460, 91.820, 92.400, 91.820, 91.820, 91.750, 91.920, 91.800, 91.110, 91.490, 91.410, 92.180, 91.600, 92.140, 91.570, 91.440, 91.930, 92.020, 90.990, 91.940, 91.630, 92.040, 92.150, 92.060, 91.840, 91.880, 91.430, 92.100, 91.430, 91.380, 91.310, 91.970, 91.590, 92.000, 92.000, 92.500, 92.140, 91.510, 91.470, 92.030, 91.500, 92.010, 91.900, 91.900, 91.250, 91.810, 91.480, 91.640, 91.390, 91.760, 92.040, 91.930, 91.800, 91.830, 91.820, 91.750, 92.150, 91.610, 91.520, 92.010, 90.290, 91.900, 91.840, 91.550, 91.500, 92.240, 91.200])

emv = ExponentialMovingAverage(0.95)
for val in accu:
    emv.push(val)
smoothed_accu = np.array(emv.get())
smoothed_delta = smoothed_accu[5:smoothed_accu.size]-smoothed_accu[:(smoothed_accu.size-5)]

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
plt.plot(smoothed_delta)
plt.ylim([-.5,.5])
plt.savefig('smoothed_delta.pdf')
