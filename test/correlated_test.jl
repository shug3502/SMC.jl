#test implementation of pmcmc and correlated pseudo marginal method

using SMC, Test, Random, LinearAlgebra, Plots, Distributions
using MCMCChains, StatsPlots, Distributed
#theme(:ggplot2);

Random.seed!(125)
trueValues = [0.9, 0.98] 
th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, trueValues[1], trueValues[2], 0.775, 2.0)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1.0, 0, 0]
y0 = [0.9, 0]

### testing generation from armond model

Random.seed!(123)
K=150
(states, observations) = generate(armondhmmSimple, x0, y0, K)
plot(th.dt*(1:K), transpose(observations))
savefig("../plots/obs.png")
@test 1>0

dimParams = 2
numRandoms = 2
#paramProposal = [x -> Beta(2.5,1), x -> Beta(2,1)]
paramProposal = [x -> TruncatedNormal(x,0.05,0,1),
                 x -> TruncatedNormal(x,0.05,0,1)]
priors = [Beta(2.5,1), Beta(2,1)]
numIter = 1000
nChains = 4
N = 16
c = zeros(numIter,dimParams,nChains)
actRate = zeros(nChains)
@time for i=1:nChains
  c[:,:,i], actRate[i] = correlated(observations, priors,
         paramProposal, dimParams, numRandoms, numIter=numIter, N=N)
end
@test size(c,1)==numIter
@test sum(actRate)/nChains > 10^-3
println(sum(c,dims=1)./numIter)

chn = Chains(c)
Rhat = gelmandiag(chn)
println(Rhat)

# visualize the MCMC simulation results
p1 = plot(chn)
savefig("../plots/chains.png")
p2 = plot(chn, colordim = :parameter)
savefig("../plots/params.png")
@test norm(sum(sum(c,dims=3),dims=1)./(numIter*nChains) .- trueValues) < 0.2
