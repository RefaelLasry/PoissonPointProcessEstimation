# Estimation of Poisson process
This project has three parts. The goal is to demonstrate a tool for estimating parameters where the MLE method may not
be carried out. 

One example where MLE is not simple and our demo will focus on is the Poisson point process

## Part 1: estimation of Poisson with Maximum Likelihood Estimation
The MLE estimator for the Poisson distribution is quite simple - the mean.

In part 1, we demonstrate generating random variables followed by the estimation. As we can see that the estimated 
parameter is close to the same parameter that generates the observations. 

part 1 is at poissonmle.py

## Part 2: estimation of Poisson with Maximum Likelihood Estimation
A powerful tool that one can use to obtain the MLE is autograd. Instead of asking what values get the maximum likelihood
by applying natural logarithm, compute the derivative(s), compare to zero, and extract "a close solution" for the
estimators. The autograde allows us to create the likelihood function and let the program automatically find the
derivative(s) and the values that get the maximum likelihood.  

you can read about the autograd:
https://pytorch.org/docs/stable/notes/autograd.html

In part 2, we demonstrate this process with three steps. In the first step, we generate random variables from a given
Poisson parameter. In the second step, we estimate the Poisson parameter with the famous MLE estimator. In the third and
last step, we estimate the Poisson parameter with autograd. 

The methods get the same results (reasonably close) as the given parameters. This autograd process provides us with a
tool to estimate parameters. It might be helpful for cases where algebra can't 
do the job.

part 2 is at poissonautomaticgrad.py

## Part 3: estimation of Poisson point process with autograd
For the Poisson point process we choose a parameter that derived from the following equation (time*c + d). Where c 
represents a decay/acceleration over time, and d represents "long time average" intensity. 

Our example builds with two steps, generating samples according to the equation and estimating the process with autograd.
The given parameters, and the estimated parameters are reasonably close. 
Q.E.D.

part 3 is at poissonpointprocess.py