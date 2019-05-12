using SMC, Test, Random, LinearAlgebra, Plots

Random.seed!(125)

th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, 0.6, 0.9, 0.775, 3.4)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1.0, 0, 0]
y0 = [0.9, 0]

### testing generation from armond model

Random.seed!(123)
K=120
N=200
(states, observations) = generate(armondhmmSimple, x0, y0, K)
@test norm(states[1,:]) < K
@test sum(states[2,:])/K < 5
@test maximum(states[3,:]) <= 1
@test minimum(states[3,:]) >= 0

#display(plot(th.dt*(1:K), observations[2,:]))

Random.seed!(155)
#provide exact initial condition x0 here
prop = bootstrapprop(armondhmmSimple,x0,transll)
auxprop = auxiliaryprop(armondhmmSimple, x0, approxtrans, approxll)

@time (psf, ess, ev) = particlefilter(hmm, observations, N, prop)
println(ess)
@time (psf, ess, ev) = particlefilter(hmm, observations, N, auxprop)
println(ess)
