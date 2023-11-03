This is code for unit testing FkMD, using a noisy Lorenz system in 3+47 dimensions (47 noise dimensions).

TO-DO: DESIGN UNIT TEST USING SMALLER SAMPLE SIZE!

One of the .mat files contains the underlying data set.

The other .mat files provide the values of Xi, Lam, V, obs_ref, obs_inf, and M after 1 and 2 iterations.

For the unit test, let's compare: (1) Phi_x * V', (2) Xi * Lam * Xi^(-1), (3) mean(abs(obs_inf-obs_ref)./abs(obs(ref))), and (4) M, across two iterations.

The .m files run FkMD on the noisy Lorenz system, using curvature or flux to get M.

Note: the "M" here is actually the matrix square root of M!
