#test implementation of pmcmc and correlated pseudo marginal method

using SMC, Test, Random, LinearAlgebra, Plots, Distributions

Random.seed!(125)

th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, 0.6, 0.95, 0.775, 3.4)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1.0, 0, 0]
y0 = [0.9, 0]

### testing generation from armond model

Random.seed!(123)
K=120
N=200
(states, observations) = generate(armondhmmSimple, x0, y0, K)
plot(transpose(observations), layout=2)
savefig("obs.png")
@test 1>0

dimParams = 2
numRandoms = 2
#paramProposal = [x -> Beta(2.5,1), x -> Beta(2,1)]
paramProposal = [x -> TruncatedNormal(x,0.05,0,1),
                 x -> TruncatedNormal(x,0.05,0,1)]
priors = [Beta(2.5,1), Beta(2,1)]
numIter = 20000
@time (c, actRate) = correlated(observations, priors,
         paramProposal, dimParams, numRandoms, numIter=numIter, N=16)

@test size(c,2)==numIter
@test actRate > 10^-3
println(sum(c,dims=2)./numIter)
display(plot(1:numIter, transpose(c[:,:]),layout=2))
savefig("test.png")
@test norm(sum(c,dims=2)./numIter .- [0.6, 0.9]) < 0.2
