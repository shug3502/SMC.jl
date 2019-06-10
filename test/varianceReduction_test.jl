using SMC, Test, Random, Distributions, Plots
Random.seed!(125)
using Statistics: median, var
th = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, 0.9, 0.98, 0.775, 2.0)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1, 0, 0]
y0 = [0.9, 0]

Random.seed!(123)
K = 150
(states, observations) = generate(armondhmmSimple, x0, y0, K)

###################

th1 = [0.9 0.98]
th2 = [0.89 0.97]

nRepeats = 10
R = zeros(nRepeats,3)
@time for i = 1:nRepeats
  R[i,1] = computeLikRatio(th1,th2,observations,N=64,rho=0.99,resampler=resample)
end

@time for i = 1:nRepeats
  R[i,2] = computeLikRatio(th1,th2,observations,N=64,rho=0.99,resampler=sortedresample)  
end

@time for i = 1:nRepeats
  R[i,3] = computeCoupledLikRatio(th1,th2,observations,N=64,rho=0.99)
end

for j=1:3
println("Likelihood ratio of $th1 vs $th2 is ",median(R[:,j])," \u00B1 ", sqrt(var(R[:,j])))
end

#@test var(R[:,1]) > var(R[:,2]) #using correlated noise as above should reduce the variance
p1 = histogram(R, layout = 3);
display(p1)

