This repository runs the Lorenz96 experiment described in "Featuring Koopman mode decomposition, with applications to cancer cell signaling" by D. Aristoff, J. Copperman, N. Mankovich, and A. Davies. 

A description of the experiment can be found at https://arxiv.org/abs/2312.09146v2.

"lorenz96_sim.m" generates the data file lorenz96_data.mat.

"FKMD_sim.m" runs FKMD on the Lorenz96 data. This generates a data file for each iteration.

"FKMD_plot.m" creates the plots that appear in Figures 1-2.

To reproduce Figs. 1-2: Run "FKMD_sim.m" with N=10^6, R=7500, delay=100, noise=1, and iters=5 to get the FKMD data; run "FKMD_sim.m" with N=10^6+99, R=7500, delay=1, noise=0, and iters=1 to get the ordinary KMD data. To plot FKMD results, open "FKMD_N1000000_R7500_l100_noise1_iter5_steps100.mat", and run "FKMD_plot.m". Similar steps can be used to plot the regular KMD results.
