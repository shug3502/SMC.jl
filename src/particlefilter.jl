export particlefilter

function particlefilter(hmm::HMM, observations::Matrix{Float}, N::Int,
                        proposal::Proposal;
                        resampling::Function=multinomialresampling,
                        essthresh::Float=0.5, u=nothing
                        )::Tuple
    #can choose to pass random numbers for particle filter to use via u
    K = size(observations, 2)
    u = isnothing(u) ? rand(K*(N+1)) : u #draw however many random numbers are needed
    @assert length(u)>2 #make sure enough random numbers to resample
    # particle set filter (storage)
    psf = ParticleSet(N, hmm.dimx, K)
    ess = zeros(K)
    ev = 0
    (p1,e1) = resample( Particles(
                            [proposal.mu0 + proposal.noise() for i in 1:N],
                            ones(N)/N),
                        essthresh, resampling, 0, u[1])
    # store
    psf.p[1] = p1
    ess[1]   = e1

    for k=2:K
        pkm1 = psf.p[k-1]
        obsk = observations[:,k]
        obskm1 = observations[:,k-1] #only used properly for aux filter

        logak = zeros(N)
        xk    = similar(pkm1.x)
        # sample (BOOTSTRAP)
        for i in 1:N
            xk[i]    = proposal.mean(k, pkm1.x[i], obskm1, obsk, u[(k-1)*N+i]) + proposal.noise(k,u[(k-1)*N+i]) #pass a single random number to the fwd simulation
            logak[i] = hmm.transloglik(k, pkm1.x[i], xk[i]) +
                        hmm.obsloglik(k, obskm1, obsk, xk[i]) -
                        proposal.loglik(k, pkm1.x[i], obskm1, obsk, xk[i])
        end
        Wk  = log.(pkm1.w) + logak
        ev += sum(logak)/N #to compute the likelihood/evidence for p(y|c)
        Wk .-= minimum(Wk) # try to avoid underflows
        wk  = exp.(Wk)
        wk /= sum(wk)
        (pk, ek) = resample(Particles(xk,wk), essthresh, resampling, 0, u[K*(N+1)-(k-2)])

        psf.p[k] = pk
        ess[k]   = ek
    end
    (psf, ess, ev)
end
