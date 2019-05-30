export particlefilter,
       parallelParticlefilter,
       f_pmap

function particlefilter(hmm::HMM, observations::Matrix{Float}, N::Int,
                        proposal::Proposal;
                        resampling::Function=multinomialresampling,
                        essthresh::Float=0.5, u=nothing,
                        resampler::Function=resample 
                        )::Tuple
    #can choose to pass random numbers for particle filter to use via u
    K = size(observations, 2)
    u = isnothing(u) ? rand(K*(N+1)) : u #draw however many random numbers are needed
    @assert length(u)>2 #make sure enough random numbers to resample
    # particle set filter (storage)
    psf = ParticleSet(N, hmm.dimx, K)
    ess = zeros(K)
    ev = 0
    (p1,e1) = resampler( Particles(
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
        (pk, ek) = resampler(Particles(xk,wk), essthresh, resampling, 0, u[K*(N+1)-(k-2)])

        psf.p[k] = pk
        ess[k]   = ek
    end
    (psf, ess, ev)
end

function parallelParticlefilter(hmm::HMM, observations::Matrix{Float}, N::Int,
                        proposal::Proposal;
                        resampling::Function=multinomialresampling,
                        essthresh::Float=0.5, u=nothing
                        )::Tuple
#=
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

    logak = SharedArray{Float,1}(N)
    xkShared = SharedArray{Int,2}(hmm.dimx,N)
    for k=2:K
        pkm1 = psf.p[k-1]
        obsk = observations[:,k]
        obskm1 = observations[:,k-1] #only used properly for aux filter
        #logak = zeros(N)
        #xk    = similar(pkm1.x)
        # sample (BOOTSTRAP)

        for i in 1:N
            xk[i]    = proposal.mean(k, pkm1.x[i], obskm1, obsk, u[(k-1)*N+i]) + proposal.noise(k,u[(k-1)*N+i]) #pass a single random number to the fwd simulation
            logak[i] = hmm.transloglik(k, pkm1.x[i], xk[i]) +
                        hmm.obsloglik(k, obskm1, obsk, xk[i]) -
                        proposal.loglik(k, pkm1.x[i], obskm1, obsk, xk[i])
        end
        xk = f_pmap(i -> proposal.mean(k, pkm1.x[i], obskm1, obsk, u[(k-1)*N+i]) + proposal.noise(k,u[(k-1)*N+i]), 1:N) #parallelize
        logak = f_pmap(i -> hmm.transloglik(k, pkm1.x[i], xk[i]) +
                        hmm.obsloglik(k, obskm1, obsk, xk[i]) -
                        proposal.loglik(k, pkm1.x[i], obskm1, obsk, xk[i]), 1:N)

@distributed (+) for i = 1:N
  xkShared[:,i] = [0 1 0 0 ] #proposal.mean(k, pkm1.x[i], obskm1, obsk, u[(k-1)*N+i]) + proposal.noise(k,u[(k-1)*N+i]);
  logak[i] = i^2 #= hmm.transloglik(k, pkm1.x[i], xkShared[:,i]) +
                        hmm.obsloglik(k, obskm1, obsk, xkShared[:,i]) -
                        proposal.loglik(k, pkm1.x[i], obskm1, obsk, xkShared[:,i]) =#
end
        Wk  = log.(pkm1.w) + logak
        ev += sum(logak)/N #to compute the likelihood/evidence for p(y|c)
        Wk .-= minimum(Wk) # try to avoid underflows
        wk  = exp.(Wk)
        wk /= sum(wk)
xk = similar(pkm1.x)
    for j=1:N
        xk[j] = xkShared[:,j]
    end
        (pk, ek) = resample(Particles(xk,wk), essthresh, resampling, 0, u[K*(N+1)-(k-2)])

        psf.p[k] = pk
        ess[k]   = ek
    end
    (psf, ess, ev)
end
=#
nCores = 8
M = Int(max(1,N/nCores))
println(M)

function unpackPF(M)
(psf, ess, ev) = particlefilter(hmm, observations, M,
                        proposal;
                        resampling=resampling,essthresh=essthresh,u=u)
return ev
end

evSum = @distributed (+) for i= 1:8 
unpackPF(M);
end
return evSum/nCores
end

# The arguments are: 1) a function 'f' and 2) a list with the input.
function f_pmap(f, lst)
    np = nprocs()            # Number of processes available.
    n  = length(lst)         # Number of elements to apply the function.
    results = Array{Any}(undef, n) # Where we will write the results. As we do not know
                             # the type (Integer, Tuple...) we write "Any"
    i = 1
    nextidx() = (idx = i; i += 1; idx) # Function to know which is the next work item.
                                       # In this case it is just an index.
    @sync begin # See below the discussion about all this part.
        for p = 1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[idx] = remotecall_fetch(f, p, lst[idx])
                    end
                end
            end
        end
    end
    results
end
