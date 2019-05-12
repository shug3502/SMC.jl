using SMC, Test, Random, Distributions, Plots

Random.seed!(125)

parameterProp = [Beta(2.5,1), Beta(2,1)] #these are the same as the prior
priors = nothing #use defaults


th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, 0.6, 0.9, 0.775, 3.4)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1.0, 0, 0]
y0 = [0.9, 0]
Random.seed!(123)
K=120
(states, observations) = generate(armondhmmSimple, x0, y0, K)

N=20
Nthet = 10
thin = 16
out = smc2(observations, Nthet,
           N, thin, priors,parameterProp)
