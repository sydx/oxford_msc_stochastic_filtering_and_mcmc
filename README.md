# Bayesian methods for solving estimation and forecasting problems in the high-frequency trading environment: code repository

The code repository created as part of Dr Paul Bilokon's MSc at the University of Oxford: Bayesian methods for solving estimation and forecasting problems in the high-frequency trading environment.

## Abstract

We examine modern stochastic filtering and Markov chain Monte Carlo (MCMC) methods and consider their applications in finance, especially electronic trading.

Stochastic filtering methods have found many applications, from Space Shuttles to self-driving cars. We review some classical and modern algorithms and show how they can be used to estimate and forecast econometric models, stochastic volatility and term structure of risky bonds. We discuss the practicalities, such as outlier filtering, parameter estimation, and diagnostics.

We focus on one particular application, stochastic volatility with leverage, and show how recent advances in filtering methods can help in this application: kernel density estimation can be used to estimate the predicted observation, filter out outliers, detect structural change, and improve the root mean square error while preventing discontinuities due to the resampling step.

We then take a closer look at the discretisation of the continuous-time stochastic volatility models and show how an alternative discretisation, based on what we call a filtering Euler--Maruyama scheme, together with our generalisation of Gaussian assumed density filters to arbitrary (not necessarily additive) correlated process and observation noises, gives rise to a new, very fast approximate filter for stochastic volatility with leverage. Its accuracy falls short of particle filters but beats the unscented Kalman filter. Due to its speed and reliance exclusively on scalar computations this filter will be particularly useful in a high-frequency trading environment.

In the final chapter we examine the leverage effect in high-frequency trade data, using last data point interpolation, tick and wall-clock time and generalise the models to take into account the time intervals between the ticks.

We use a combination of MCMC methods and particle filtering methods. The robustness of the latter helps estimate parameters and compute Bayes factors. The speed and precision of modern filtering algorithms enables real-time filtering and prediction of the state.


