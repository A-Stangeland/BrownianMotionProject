import numpy as np
import numpy.random as npr
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def test_plot1():
    x = np.zeros(2)
    point = plt.plot(x, 'b.')[0]
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    for i in range(1000):
        u = npr.normal(0,1,2)
        c = npr.uniform(0,1,3)#npr.randint(0, 256, 3)
        x += u
        print(x)
        plt.plot(x[0], x[1], '.', color=c)
        #point.set_data(x)
        #point.set_color(c)
        plt.pause(0.01)
    plt.show()

def test_plot2():
    U = npr.normal(0, 1, (1000, 2))
    #U = np.ones((10,2))
    X = np.cumsum(U, axis=0)
    print(X)
    plt.plot(X[:,0], X[:,1])
    plt.show()
    print(X.shape)

def wiener_path(seed=None):
    rng = default_rng(seed)
    N = 1000
    T = 1
    dt = T / N
    t = np.linspace(0, 1, N+1)
    X = rng.standard_normal(N)
    W = np.zeros(N+1)
    W[1:] = np.cumsum(X*np.sqrt(dt))
    #fig, axs = plt.subplots(2,2)
    #i = np.arange(0, N+1, 500)
    #ax[0,0].plot(t[index], W[index], color='black')

    m, M = np.min(W), np.max(W)
    for i, k in enumerate((250, 100, 50, 10, 5, 1)):
        index = np.arange(0, N+1, k)
        #index = np.linspace(0, N+1, )
        print(index)
        matplotlib.rcParams.update({'font.size': 8})
        ax = plt.subplot(3, 2, i+1)
        ax.plot(t[index], W[index], 'k', lw=0.7)
        ax.set_ylim([1.1*m, 1.1*M])
        ax.set_title('n=%d' % (N // k))
        ax.set_yticks([])
        #ax.set_xticks([0, 1], minor=False)
        #ax.set_xticks([0.2, 0.4, 0.6, 0.8], minor=True)
        #ax.set_xlabels(['0', 'T'])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds((0, 1))
        #plt.axis('off')
    plt.tight_layout()
    plt.show()


def KLBM(N, T=1):
    npr.seed(271)
    t = np.linspace(0, T, 1000)
    k = (2*np.arange(N) + 1)[:,None]
    Z = npr.normal(size=(N,1))
    phi = 2*np.sqrt(2*T)*np.sin(k*np.pi*t[None,:]/(2*T))/(k*np.pi)
    W = Z * phi
    #plt.figure(figsize=(20,10))
    line = plt.plot([],[], lw=0.7)[0]
    text = plt.text(0.8, 0.3, '', fontweight='bold', bbox=dict(facecolor='gold', alpha=0.5))
    plt.xlim([0,T])
    plt.ylim([-2, 0.5])
    for i in range(1, N):
        line.set_data(t, np.sum(W[:i,:], axis=0))
        text.set_text('n = %s' % i)
        plt.pause(1/i)
    plt.show()

def GBM(N, T=1, x=1, r=1.5, sig=0.1):
    npr.seed(271)
    t = np.linspace(0, T, 1000)
    k = (2*np.arange(N) + 1)[:,None]
    Z = npr.normal(size=(N,1))
    phi = 2*np.sqrt(2*T)*np.sin(k*np.pi*t[None,:]/(2*T))/(k*np.pi)
    W = np.sum(Z * phi, axis=0)
    S = x * np.exp((r-sig**2/2)*t + sig*W)


    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.25)
    line = plt.plot(t, S)[0]
    plt.xlim([0,1])
    plt.ylim([0, 1.5+r**2])
    axcolor = 'lightgoldenrodyellow'
    axr = plt.axes([0.2, 0.1, 0.3, 0.03], facecolor=axcolor)
    axsig = plt.axes([0.6, 0.1, 0.3, 0.03], facecolor=axcolor)
    rslider = Slider(axr, '$r$', 1, 5, valinit=1.)
    sigslider = Slider(axsig, r'$\sigma$', 0.1, 2, valinit=.5)

    def update(val):
        r = rslider.val
        sig = sigslider.val
        S = x * np.exp((r-sig**2/2)*t + sig*W)
        line.set_data(t, S)
        ax.set_ylim([0,1.5+r**2])
        fig.canvas.draw_idle()
    
    rslider.on_changed(update)
    sigslider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        rslider.reset()
        sigslider.reset()
        npr.seed()
        Z = npr.normal(size=(N,1))
        W = np.sum(Z * phi, axis=0)
        r = rslider.val
        sig = sigslider.val
        S = x * np.exp((r-sig**2/2)*t + sig*W)
        line.set_data(t, S)
        fig.canvas.draw_idle()
        

    button.on_clicked(reset)

    plt.show()

def random_walk():
    rng = default_rng()
    N = 1000
    T = 1
    dt = T / N
    t = np.linspace(0, 1, N+1)
    X = rng.integers(2, size=N)*2 - 1
    W = np.zeros(N+1)
    W[1:] = np.cumsum(X*dt)

    m, M = np.min(W), np.max(W)
    for i, k in enumerate((4, 10, 20, 100, 200, 1000)):
        matplotlib.rcParams.update({'font.size': 8})
        ax = plt.subplot(3, 2, i+1)
        index = np.arange(0, N+1, N//k)
        print(index)
        ax.plot(t[index], W[:k+1], color='black', lw=0.7)
        #ax.set_ylim([1.1*m, 1.1*M])
        ax.set_title('n=%d' % k)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds((0, 1))
        #plt.axis('off')
    plt.tight_layout()
    plt.show()

def BM_spectral_function(kmax, T=1, seed=None):
    rng = default_rng(seed)
    k = (2*np.arange(kmax) + 1)[:,None]
    Z = rng.standard_normal((kmax,1))
    phi = lambda t: 2*np.sqrt(2*T)*np.sin(k*np.pi*t[None,:]/(2*T))/(k*np.pi)
    W = lambda t: np.sum(Z*phi(t), axis=0)
    return W, k[:,0]/(4*T), np.abs(Z*2*np.sqrt(2*T)/(k*np.pi))[:,0]


def brownian_noise():
    BM_function, freq, amp = BM_spectral_function(1000)
    N = 1000
    T = 1
    t = np.linspace(0, T, N+1)
    Wt = BM_function(t)
    plt.subplot(1,2,1)
    plt.plot(t, Wt, 'k', lw=0.7)

    plt.subplot(1,2,2)
    logamp = np.log(amp)
    logamp -= np.min(logamp)
    print(logamp)
    plt.bar(freq, logamp, width=freq[1]-freq[0])
    #plt.semilogy(freq, amp)
    
    plt.show()

def spectral_BM():
    N = 2000
    K = N // 2
    T = 1
    t = np.linspace(0, T, N)
    k = (2*np.arange(K) + 1)[:,None]
    Z = npr.normal(size=(K,1))
    phi = 2*np.sqrt(2*T)*np.sin(k*np.pi*t[None,:]/(2*T))/(k*np.pi)
    W = np.cumsum(Z * phi, axis=0)
    print(W.shape)
    m, M = np.min(W), np.max(W)
    matplotlib.rcParams.update({'font.size': 8})
    for i, n in enumerate([4, 10, 20, 100, 200, 1000]):
        ax = plt.subplot(3, 2, i+1)
        ax.plot(t, W[n-1], 'k', lw=0.7)
        ax.set_ylim([1.1*m, 1.1*M])
        ax.set_title('n=%d' % n)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds((0, 1))
    plt.tight_layout()
    plt.show()

def triang(t):
    T = np.clip(t, 0, 1)
    T = 1 - 2*np.abs(0.5 - T)
    return T

def wavelet_construction():
    N = 100
    n = np.arange(N)
    j = np.zeros_like(n)
    j[2:] = np.log2(n[2:]).astype(int)
    print(j)
    j2 = 2.**j
    k = n - j2
    t = np.linspace(0, 1, 2*N)
    T = j2[None,:] * t[:,None] - k[None, :]
    print(T.shape)


def nowhere_diff():
    rng = default_rng()
    N = 10000
    T = 0.001
    dt = T / N
    t = np.linspace(0, 1, N+1)
    X = rng.standard_normal(N)
    W = np.zeros(N+1)
    W[1:] = np.cumsum(X*np.sqrt(dt))
    K = 100
    dW = np.zeros(K)
    for k in range(1, K+1):
        dW[k-1] = np.mean(np.abs(W[:-k:k] - W[k::k])) / (k*dt)
        #dW[k-1] = np.min(np.abs(W[:-k:k] - W[k::k])) / (k*dt)
        #dW[k-1] = np.abs(W[100+k] - W[100]) / (k*dt)
    
    deltat = dt*np.arange(1,K+1)

    ax = plt.subplot()
    ax.plot(deltat, dW, 'k', lw=0.7)
    #ax.semilogy(deltat, dW, 'k', lw=0.7)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$| B_{t + \Delta t} - B_t | / \Delta t$')
    #ax.set_ylabel(r'$\frac{| B_{t + \Delta t} - B_t |}{\Delta t}$')
    ticks = [k*K*dt/5 for k in range(6)]
    #plt.xticks(ticks=ticks, labels=['{:.0e}'.format(x) for x in ticks])
    plt.tight_layout()
    plt.show()

def nowhere_diff2():
    rng = default_rng()
    K = 1000
    k = np.linspace(-10, -1, K)
    dt = 10**k
    n = 100
    B = np.mean(np.abs(rng.standard_normal((n, K))), axis=0) * np.sqrt(dt)
    dB = B / dt

    ax = plt.subplot()
    ax.semilogy(dt, dB, 'k', lw=0.7)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$| B_{t + \Delta t} - B_t | / \Delta t$')
    #ax.set_ylabel(r'$\frac{| B_{t + \Delta t} - B_t |}{\Delta t}$')
    #ticks = [k*K*dt/5 for k in range(6)]
    #plt.xticks(ticks=ticks, labels=['{:.0e}'.format(x) for x in ticks])
    plt.tight_layout()
    plt.show()

def brownian_monster():
    rng = default_rng()
    N = 1000
    K = 500
    T = 1
    dt = T / N
    t = np.linspace(0, 1, N+1)
    X = rng.standard_normal((K, N))
    W = np.zeros((K,N+1))
    W[:,1:] = np.cumsum(X*np.sqrt(dt), axis=1)
    
    plt.style.use('dark_background')
    for k in range(K):
        plt.plot(W[k], t[::-1], 'w', lw=0.3)

    plt.plot((-0.1, 0.1),(0.9, 0.9), 'r.')
    x = np.linspace(-0.3, 0.3, 100)
    plt.plot(x, -0.005*np.cos(100*x) + 0.82 + 0.5*x**2, 'r', lw=0.3)


    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def brownian2d():
    rng = default_rng()
    N = 10000
    K = 10
    T = 1
    dt = T / N
    X1 = rng.standard_normal((K, N))
    X2 = rng.standard_normal((K, N))
    W1 = np.zeros((K,N+1))
    W1[:,1:] = np.cumsum(X1*np.sqrt(dt), axis=1)
    W2 = np.zeros((K,N+1))
    W2[:,1:] = np.cumsum(X2*np.sqrt(dt), axis=1)
    
    #plt.style.use('dark_background')
    for k in range(K):
        plt.plot(W1[k], W2[k], lw=0.3)


    plt.axis('off')
    plt.tight_layout()
    plt.show()

def brownian_particles():
    rng = default_rng()
    xlim = 1.61803398875
    ylim = 1
    N = 10000
    K = 10
    T = 1
    dt = T / N
    X1 = rng.standard_normal((N, K))
    X2 = rng.standard_normal((N, K))
    W1 = np.zeros((N+1, K))
    W2 = np.zeros((N+1, K))
    W1[0] = rng.uniform(-xlim, xlim, size=K)
    W2[0] = rng.uniform(-ylim, ylim, size=K)
    
    for n in range(N):
        W1[n+1] = np.where(np.abs(W1[n] + np.sqrt(dt)*X1[n]) < xlim, W1[n] + np.sqrt(dt)*X1[n], W1[n] - np.sqrt(dt)*X1[n])
        W2[n+1] = np.where(np.abs(W2[n] + np.sqrt(dt)*X2[n]) < ylim, W2[n] + np.sqrt(dt)*X2[n], W2[n] - np.sqrt(dt)*X2[n])
    #W1 = np.sqrt(dt) * W1
    #W2 = np.sqrt(dt) * W2

    scale = 5
    plt.style.use('dark_background')

    plt.figure(figsize=((scale*xlim, scale*ylim)))
    points = plt.plot([], [], '.')[0]
    lines = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k in range(K):
        lines.append(plt.plot([], [], color=colors[k], lw=0.2)[0])
    a = 1
    plt.xlim([-a*xlim, a*xlim])
    plt.ylim([-a*ylim, a*ylim])
    plt.axis('off')
    for n in range(N):
        points.set_data(W1[n], W2[n])
        for k in range(K):
            lines[k].set_data(W1[:n, k], W2[:n, k])
        plt.pause(0.01)

    plt.show()

def brownian_path():
    rng = default_rng(56)
    N = 10000
    K = 1
    T = 1
    dt = T / N
    t = np.linspace(0, 1, N+1)
    X = rng.standard_normal((K, N))
    W = np.zeros((K,N+1))
    W[:,1:] = np.cumsum(X*np.sqrt(dt), axis=1)
    
    plt.figure(figsize=(5*1.61803398875,5))
    for k in range(K):
        plt.plot(t, W[k], 'k', lw=0.5)

    plt.ylim([-1.1, 1.6])
    plt.yticks(np.arange(-1, 2, 0.5))
    plt.tight_layout()
    plt.show()

brownian_particles()