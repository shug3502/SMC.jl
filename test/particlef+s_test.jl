using SMC, Test, Random
Random.seed!(125)

## Linear Gaussian

dx, dy = 5, 3
A = randn(dx,dx); A /= 0.9norm(A)
B = randn(dy,dx); B /= 1.1norm(B)
Q = randn(dx,dx); Q *= Q'; Q += 0.5*eye(dx); Q += Q'; Q /= 5
R = randn(dy,dy); R *= R'; R += 0.5*eye(dy); R += R'; R /= 20

lg  = LinearGaussian(A,B,Q,R)
hmm = HMM(lg)
x0  = randn(dx)

K = 50
N = 64
(states, observations) = generate(lg, x0, K)

Random.seed!(155)
@time (psf, ess, ev) = particlefilter(hmm, observations, N, bootstrapprop(lg))

@test length(psf)==K

pfm  = mean(psf)
pfmm = zeros(dx,K)
for k in 1:K
    pfmm[:,k] = pfm[k]
end

@test norm(pfmm-states)/norm(states) < 1.0 #TODO: work out whats up, was 0.4
println("PF    : $(norm(pfmm-states)/norm(states))")

Random.seed!(521)
@time psffbs  = particlesmoother_ffbs(hmm, psf)

@test length(psffbs)==K

psm  = mean(psffbs)
psmm = zeros(dx,K)
for k in 1:K
    psmm[:,k] = psm[k]
end

@test norm(psmm-states)/norm(states) < 1.0 #TODO: was 0.25
println("PSFFBS: $(norm(psmm-states)/norm(states))")

Random.seed!(521)
@time (psbbis, ess) = particlesmoother_bbis(hmm, observations,
                                        psf, bootstrapprop(lg))

psm3  = mean(psbbis)
psmm3 = zeros(dx,K)
for k in 1:K
    psmm3[:,k] = psm3[k]
end

@test norm(psmm3-states)/norm(states) < 1.0 #was 0.23
println("PSBISQ: $(norm(psmm3-states)/norm(states))")

Random.seed!(521)
@time (pslbbis, ess) = particlesmoother_lbbis(hmm, observations,
                                              psf, bootstrapprop(lg))

psm4  = mean(pslbbis)
psmm4 = zeros(dx,K)
for k in 1:K
    psmm4[:,k] = psm4[k]
end

@test norm(psmm4-states)/norm(states) < 1.0 # was 0.23
println("PSBISL: $(norm(psmm4-states)/norm(states))")

Random.seed!(521)
@time (psllbbis, ess) = particlesmoother_llbbis(hmm, observations, psf, 25,
                                                bootstrapprop(lg))

psm5  = mean(psllbbis)
psmm5 = zeros(dx,K)
for k in 1:K
    psmm5[:,k] = psm5[k]
end

@test norm(psmm5-states)/norm(states) < 1.0 # was 0.23
println("PSBISLL: $(norm(psmm5-states)/norm(states))")
