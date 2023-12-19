This repository runs the Lorenz96 experiment described in "Featuring Koopman mode decomposition, with applications to cancer cell signaling" by D. Aristoff, J. Copperman, N. Mankovich, and A. Davies. 

A description of the experiment can be found at https://arxiv.org/abs/2312.09146v2.

"lorenz96_sim" generates the data file lorenz96_data.mat.

"FKMD_sim" runs FKMD on the lorenz data.

"FKMD_plot" plots the figures in https://arxiv.org/abs/2312.09146v2.

Note: to reproduce the FKMD data in Figs. 1-2, use N=10^5, R=7500, delay=100, noise=1, iters=8.
To reproduce the KMD data in Fig. 1, use N=10^5+100, R=7500, delay=1, noise=0, iters=1.

