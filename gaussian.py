import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

N = 5

means = np.random.randn(N,2) * 10 + np.array([100, 200])
stdev = np.random.randn(N,2) * 10 + 30
count = np.int64(np.int64(np.random.randn(N,2) * 10000 + 50000))

a = [np.hstack([np.random.randn(count[i,j]) * stdev[i,j] + means[i,j] for j in range(2)]) for i in range(N)]

limmin = min(np.percentile(x, 0.1) for x in a)
limmax = max(np.percentile(x, 99.9) for x in a)
ls = np.linspace(limmin, limmax, 100)

for n in range(N):
    x = a[n]
    x = x[(x > limmin)&(x < limmax)]
    kde = gaussian_kde(x)
    plt.plot(ls, kde(ls), label='data %d' % n)

plt.xlim([limmin, limmax])
plt.legend()
plt.title('data distributions')
plt.show()