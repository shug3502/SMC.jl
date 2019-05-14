using SMC, Test, Random, LinearAlgebra, Plots
using Statistics: cor
Random.seed!(125)

th = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, 0.9, 0.98, 0.775, 2.0)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1.0, 0, 0]
y0 = [0.9, 0]

### testing generation from armond model

Random.seed!(123)
K=500
N=100
(states, observations) = generate(armondhmmSimple, x0, y0, K)
#plot(th.dt*(1:K), transpose(observations))
#savefig("../plots/obs.png")

@test norm(states[1,:]) < K
@test sum(states[2,:])/K < 5
@test maximum(states[3,:]) <= 1
@test minimum(states[3,:]) >= 0
@test cor(states[1,:],states[2,:]) < 0 #states anti-correlated
@test cor(observations[1,:],observations[2,:]) > 0.5 #trajectories correlated

Random.seed!(155)
#provide exact initial condition x0 here
prop = bootstrapprop(armondhmmSimple,x0,transll)
auxprop = auxiliaryprop(armondhmmSimple, x0, approxtrans, approxll)

@time (psf, ess, ev) = particlefilter(hmm, observations, N, prop)
@time (psfaux, essaux, evaux) = particlefilter(hmm, observations, N, auxprop)
@test mean(essaux) > mean(ess) #should get better ESS from auxiliary particle filter
@test (ev < 0) & (evaux < 0) 
