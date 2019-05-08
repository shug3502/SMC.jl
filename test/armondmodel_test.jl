using SMC, Test, Random, LinearAlgebra, Plots

Random.seed!(125)

th = theta(450, 0.008, 0.025, -0.015, 0.035, 0.12/3.4, 0.06/3.4, 0.775, 3.4)
(armondhmm, transll, approxtrans, approxll) = armondModel(th)
hmm = HMM(armondhmm, transll)
x0 = [0.9, 0, 0, 1, 0, 0]
#x0  = randn(nSisters) 

### testing generation from armond model

Random.seed!(123)
K=120
N=500
(states, observations) = generate(armondhmm, x0, K)
@test norm(states[1,:]) < K
@test sum(states[2,:])/K < 5
@test maximum(states[3,:]) <= 1
@test minimum(states[3,:]) >= 0

#display(plot(th.dt*(1:K), states[2,:]))
### test transloglik
@test isapprox(transll(2,x0,[0.95,-0.01,0,0,1,0]), -Inf) #not possible in one step
@test transll(2,x0,[1.0,0.0,0,1,0,0]) < 10^3
@test isapprox(hmm.transloglik(2,x0,[0.95,-0.01,0,0,1,0]), -Inf) #not possible in one step
@test hmm.transloglik(2,x0,[1.0,0.0,0,1,0,0]) < 10^3
Random.seed!(155)
#provide exact initial condition x0 here
prop = bootstrapprop(armondhmm,x0,transll)
auxprop = auxiliaryprop(armondhmm, x0, approxtrans, approxll)

@time (psf, ess) = particlefilter(hmm, observations, N, prop)
println(ess)
@test length(psf)==K
#expect bootstrap filter to do badly
@test any([isnan(a) for a in ess])
@test isnan(minimum(ess))

@time (psf, ess) = particlefilter(hmm, observations, N, auxprop)
println(ess)
@test length(psf)==K
#expect reasonable results from aux particle filter
@test !any([isnan(a) for a in ess])
@test !isnan(minimum(ess))

pfm  = mean(psf)
#println(pfm)
#for k in 1:K
#    println(pfm[k])
#end
