import numpy as np
import numpy.random as npr
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

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

def KLBM2(N, T=1):
    #npr.seed(271)
    t = np.linspace(0, T, 1000)
    k = (2*np.arange(N) + 1)[:,None]
    #Z = npr.normal(size=(N,1))
    phi = 2*np.sqrt(2*T)*np.sin(k*np.pi*t[None,:]/(2*T))/(k*np.pi)
    W1 = phi * npr.normal(size=(N,1))
    W2 = phi * npr.normal(size=(N,1))
    #plt.figure(figsize=(20,10))
    line = plt.plot([],[], lw=0.7)[0]
    text = plt.text(1.2, 1.2, '', fontweight='bold', bbox=dict(facecolor='gold', alpha=0.5))
    a = 1.5
    plt.xlim([-a*T,a*T])
    plt.ylim([-a*T, a*T])
    for i in range(1, N):
        line.set_data(np.sum(W1[:i,:], axis=0), np.sum(W2[:i,:], axis=0))
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

def GBM2():
    rng = default_rng()
    T = 1
    N = 200
    dt = T / N
    K = 2000
    Z = rng.standard_normal((K, N))
    W = np.cumsum(Z*np.sqrt(dt), axis=1)

    t = np.linspace(0, T, N)
    x=1
    r = 2
    sig = 0.5
    S = x * np.exp((r-sig**2/2)*t[None,:] + sig*W)
    q1, q2 = 0.05, 0.95

    nbins = 100
    Heat = np.zeros((nbins, N))
    m, M = np.quantile(S, [q1, q2])
    hist_func = lambda X: np.histogram(X, bins=nbins, range=[m, M])[0]
    Heat = np.apply_along_axis(hist_func, 0, S)


    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.25)
    im = plt.imshow(Heat[::-1]**(0.5), cmap='inferno')
    plt.axis('off')
    axcolor = 'lightgoldenrodyellow'
    axr = plt.axes([0.2, 0.1, 0.3, 0.03], facecolor=axcolor)
    axsig = plt.axes([0.6, 0.1, 0.3, 0.03], facecolor=axcolor)
    rslider = Slider(axr, '$r$', 1, 5, valinit=r)
    sigslider = Slider(axsig, r'$\sigma$', 0.1, 4, valinit=sig)

    def update(val):
        r = rslider.val
        sig = sigslider.val
        S = x * np.exp((r-sig**2/2)*t[None,:] + sig*W)
        m, M = np.quantile(S, [q1, q2])
        hist_func = lambda X: np.histogram(X, bins=nbins, range=[m, M])[0]
        Heat = np.apply_along_axis(hist_func, 0, S)
        im.set_data(Heat[::-1]**(0.5))
        fig.canvas.draw_idle()
    
    rslider.on_changed(update)
    sigslider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        rslider.reset()
        sigslider.reset()
        npr.seed()
        Z = rng.standard_normal((K, N))
        W = np.cumsum(Z*np.sqrt(dt), axis=1)
        r = rslider.val
        sig = sigslider.val
        S = x * np.exp((r-sig**2/2)*t[None,:] + sig*W)
        Heat = np.zeros((nbins, N))
        m, M = np.quantile(S, [q1, q2])
        hist_func = lambda X: np.histogram(X, bins=nbins, range=[m, M])[0]
        Heat = np.apply_along_axis(hist_func, 0, S)
        im.set_data(Heat[::-1]**(0.5))
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

def heat_map():
    rng = default_rng()
    T = 1
    N = 500
    dt = T / N
    K = 20000
    Z = rng.standard_normal((K, N))
    W = np.cumsum(Z*np.sqrt(dt), axis=1)

    t = np.linspace(0, T, N)
    x=1
    r = 2
    sig = 0.5
    S = x * np.exp((r-sig**2/2)*t[None,:] + sig*W)

    nbins = 300
    Heat = np.zeros((nbins, N))
    #m, M = np.quantile(S, [0, 0.9])
    #hist_func = lambda X: np.histogram(X, bins=nbins, range=[m, M])[0]
    #Heat = np.apply_along_axis(hist_func, 0, S)


    M = np.quantile(np.abs(W), 0.99)
    hist_func = lambda X: np.histogram(X, bins=nbins, range=[-1.2*M, 1.2*M])[0]
    Heat = np.apply_along_axis(hist_func, 0, W)
    #Heat = Heat / np.max(Heat, axis=0)[None,:]
    
    plt.imshow(Heat, cmap='inferno')
    plt.axis('off')
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

def sde_integration():
    s0 = 1
    r = 1.5
    sigma = .2
    N = 1000
    K = 100
    t = np.linspace(0, 1, N)
    dt = t[1]-t[0]
    #Z = np.sqrt(dt)*npr.standard_normal((K, N))
    #W1 = np.cumsum(Z, axis=1)
    W1 = np.sqrt(dt)*np.cumsum(npr.standard_normal((K, N)), axis=1)
    S1 = s0*np.exp((r-sigma**2/2)*t +sigma*W1)

    Z = np.sqrt(dt)*npr.standard_normal((K, N))
    S2 = np.zeros_like(S1)
    S2[:,0] = s0
    for i in range(N-1):
        S2[:,i+1] = S2[:,i] + r*S2[:,i]*dt + sigma*S2[:,i]*Z[:,i]

    plot = plt.plot
    for k in range(K):
        plot(t, S1[k], color='blue', lw=0.1)
        plot(t, S2[k], color='orange', lw=0.1)
    plot(t, np.mean(S1, axis=0), '--', color='black', lw=1)
    plot(t, np.mean(S2, axis=0), '--', color='red', lw=1)
    plt.show()