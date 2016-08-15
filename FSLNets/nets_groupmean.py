import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import invgauss
from scipy.stats import t
import matplotlib.pyplot as plt


def nets_groupmean(netmats, makefigure, Nsubgroup=1):

    nf = np.sqrt(netmats.shape[1])
    n = round(nf)
    nsub = netmats.shape[0]

    # one group t-test
    grot = netmats
    DoF = nsub - 1
    if Nsubgroup > 1:
        for i in range(nsub / Nsubgroup):
            grot[i, :] = np.mean(netmats[i * Nsubgroup: (i +1) * Nsubgroup -1])
        DoF = i - 1

    Tnet = np.sqrt(grot.shape[0]) * np.mean(grot, axis=0) / np.std(grot, axis=0, ddof=1)
    Mnet = np.mean(grot, axis=0)

    Znet = np.zeros(Tnet.shape[0])
    Znet[Tnet > 0] = -invgauss.cdf(t.cdf(-Tnet[Tnet > 0], DoF))
    Znet[Tnet < 0] = -invgauss.cdf(t.cdf(-Tnet[Tnet < 0], DoF))

    Znetd = Znet
    if N == Nf:      # is netmat square....
      Znet = reshape(Znet, (N, N))
      Mnet = reshape(Mnet, (N, N))

    if makefigure > 0:
        plt.figure('position', [100, 100, 1100, 400])
        plt.subplot(1, 2, 1);
        plt.plot(Znetd);
        if N == nf: # if netmat square....
            Znetd = np.reshape(Znetd, (N, N));
        if sum(sum(abs(Znetd) - abs(Znetd.T)))<0.00000001:    # .....and symmetric
            plt.imagesc(Znetd, [-10, 10])
            colormap('jet')
            colorbar;
        plt.title('z-stat from one-group t-test');

        # scatter plot of each session's netmat vs the mean netmat
        plt.subplot(1, 2, 2);
        grot = np.matlib.repmat(mean(netmats), Nsub, 1);
        plt.scatter(netmats[:], grot[:])
        plt.title('scatter of each session''s netmat vs mean netmat')
    return Znet, Mnet

if __name__ == '__main__':
    netmats = np.genfromtxt('/home/tadlington/fNETMATS/netmats-corr.txt', delimiter=',')
    Znet, Mnet = nets_groupmean(netmats, 0)
    print hi

