This is code for unit testing FkMD, using a noisy Lorenz system in 3+47 dimensions (47 noise dimensions).

One of the .mat files contains the underlying data set.

The other .mat files provide the values of Xi, Lam, V, obs_ref, obs_inf, and M after 1 and 2 iterations.

The .m files run FkMD on the noisy Lorenz system, using curvature or flux to get M.

Note: the "M" here is actually the matrix square root of M!
