import numpy as np
import matplotlib.pyplot as plt


def quadratiq_variation(B):
    return np.cumsum(np.power(np.diff(B, axis=0, prepend=0.), 2), axis=0) #t

def main():
    n = 500  #sample points
    d = 100
    T = 1.
    times = np.linspace(0., T, n)
    dt = times[1] - times[0]
    # Bt2 - Bt1 ~ Normal with mean 0 and variance t2-t1
    dB = np.sqrt(dt) * np.random.normal(size=(n-1, d))
    B0 = np.zeros(shape=(1, d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    #fig.suptitle('Realization of n=%d Brownian motions and corresponding quadratic variation' %(n))
    ax1.plot(times, B, lw=0.3, color='black')
    ax2.plot(times, quadratiq_variation(B), lw=0.3, color='black')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()