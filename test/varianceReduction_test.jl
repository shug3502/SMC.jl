using SMC, Test, Random, Distributions, Plots
Random.seed!(125)
using Statistics: median, var
th = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, 0.9, 0.98, 0.775, 2.0)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1, 0, 0]
y0 = [0.9, 0]

Random.seed!(123)
K=150
(states, observations) = generate(armondhmmSimple, x0, y0, K)

###################
function computeLikRatio(th1::Array{Float64}, th2::Array{Float64}, observations::Array{Float64};
                         N::Int64=64, rho::Float64=0.95, resampler::Function=resample)
  numRandoms = size(observations,2)*(N+1)
  u1 = randn(numRandoms)
  u2 = randn(numRandoms)
  u3 = rho*u1 + sqrt(1-rho^2)*u2
  lik1 = runFilter(th1, cdf.(Normal(),u1), observations, x0=x0, N=N, resampler=sortedresample)
  ratio1 = exp(lik1 - runFilter(th2, cdf.(Normal(),u2), observations, x0=x0, N=N, resampler=resampler))
  ratio2 = exp(lik1 - runFilter(th2, cdf.(Normal(),u3), observations, x0=x0, N=N, resampler=resampler))
  return [ratio1, ratio2]
end

th1 = [0.9 0.98]
th2 = [0.6 0.95]

nRepeats = 100
R = zeros(nRepeats,2)
@time for i = 1:nRepeats
  R[i,1:2] = computeLikRatio(th1,th2,observations,N=64,rho=0.99,resampler=sortedresample)  
end

println("Likelihood ratio of $th1 vs $th2 is ",median(R[:,1])," \u00B1 ", sqrt(var(R[:,1])))
println("Likelihood ratio of $th1 vs $th2 is ",median(R[:,2])," \u00B1 ", sqrt(var(R[:,2])))
#@test var(R[:,1]) > var(R[:,2]) #using correlated noise as above should reduce the variance
p1 = histogram(R, layout = 2);
display(p1)

