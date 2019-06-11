export
    resample,
    sortedresample,
    maxcouplingresample,
    multinomialresampling,
    systematicresampling
"""
    resample(p::Particles, essthresh, rs, M, u)

Resamples the particle object `p` if the ess is under `essthresh`. The
resampling algorithm `rs` is for example a multinomial resampling.
"""
function resample(p::Particles, essthresh::Float=Inf,
                  rs::Function=multinomialresampling, M::Int=0,
                  u=nothing
                  )::Tuple{Particles,Array,Float}
    ess = 1.0/sum(p.w.^2)
    N   = length(p)
    M   = M>0 ? M : N
    (M != N || ess < essthresh * N) ? (rs(p, M, u)..., ess) : (p, 1:N, ess)
end

"""
    sortedresample(p::Particles, essthresh, rs, M, u)

Resamples the particle object `p` if the ess is under `essthresh`. The
resampling algorithm `rs` is for example a stratified resampling.
First sorts the particles before resampling to allow passing randomness
through without destroying correlations between the particles.
Project from binary hidden states to a line to enable sorting.
"""
function sortedresample(p::Particles, essthresh::Float=Inf,
                        rs::Function=stratifiedresampling, M::Int=0,
                        u=nothing
                        )::Tuple{Particles,Array,Float}
    ess = 1.0/sum(p.w.^2)
    N   = length(p)
    M   = M>0 ? M : N
    dimx = length(p.x[1])

    #sort before resampling
    #assume binary hidden states; to do this for real state space use transform from Hilbert Space Filling Curve
    q = zeros(N) #similar(p.x)
    twos = transpose(2 .^(0:(dimx-1)))
    for i=1:N
        q[i] = twos * p.x[i]
    end
    #compute indices and use these to sort the particles
    sortedIndx = convert(Array{Int},sortslices(hcat(q, 1:N), dims = 1, by = x -> x[1])[:,2])
    p.x = p.x[sortedIndx]
    p.w = p.w[sortedIndx]
    (M != N || ess < essthresh * N) ? (rs(p, M, u)..., ess) : (p, 1:N, ess)
end

"""
    maxcouplingresample(pkm1::Particles, pk::Particles, essthresh, rs, M, u)

Resamples the particle object `p` if the ess is under `essthresh`. The
resampling algorithm `rs` is for example a stratified resampling.
Couples between previous particles and new particles via max coupling scheme. 
See Sen et al 2018 Stat. Comput for description.
"""
function maxcouplingresample(pkm1::Particles, pk::Particles, essthresh::Float=Inf,
                        rs::Function=stratifiedresampling, M::Int=0,
                        u=nothing
                        )::Tuple{Particles,Particles,Array,Float}
    ess = 1.0/sum(pkm1.w.^2)
    N = length(pkm1)
    @assert length(pk) == N
    M   = M>0 ? M : N
    p = 0.0
    mu = zeros(N)
    mu1 = zeros(N)
    mu2 = zeros(N)
    for i=1:N
        mu[i] = min(pkm1.w[i],pk.w[i])
        p += mu[i]
        mu1[i] = pkm1.w[i] - mu[i]
        mu2[i] = pk.w[i] - mu[i]
    end
    @assert p != 0 "Should be ok in this case but should check just in case"
    mu /= p #normalize
    mu1 /= 1-p
    mu2 /= 1-p
    r = rand(M) #independent source of randomness
    Y = zeros(N)
    Z = zeros(N)
    if (M != N || ess < essthresh * N)
        coupled, an_coupled = rs(Particles(pkm1.x, mu), M, u)
        Yuncoupled, ~ = rs(Particles(pkm1.x,mu1),M,u)
        Zuncoupled, Zan_uncoupled = rs(Particles(pk.x,mu2),M,u)
        pkm1.w = ones(M)/M
        pk.w = ones(M)/M
        ancestors = zeros(M)
        for i=1:N
            (pkm1.x[i], pk.x[i]) = (r[i]<p) ? (coupled.x[i],coupled.x[i]) : (Yuncoupled.x[i], Zuncoupled.x[i])
            ancestors[i] = (r[i]<p) ? an_coupled[i] : Zan_uncoupled[i] 
        end
    else
        ancestors = 1:N
    end
    return (pkm1, pk, ancestors, ess)
end

"""
    multinomialresampling(p::Particles, M, u)

Multinomial resampling of a particles object `p`.
"""
function multinomialresampling(p::Particles, M::Int=0, u=nothing)::Particles
    #@assert isnothing(u) #passing randomness to multinomial resampling not currently implemented
    N    = length(p)
    M    = (M>0) ? M : N
    ni   = rand(Multinomial(M, p.w))
    ancestors = [j for i in 1:N for j in ones(Int,ni[i])*i]
    return Particles(p.x[ancestors], ones(M)/M), ancestors
end

"""
    systematicresampling(p::Particles, M, u)
Systematic resampling of a particles opject `p`.
Helpful when wanting to correlate randomness in successive filter evaluations.
"""
function systematicresampling(p::Particles, M::Int=0, u=nothing)::Tuple{Particles,Array}
    #based on MATLAB code from https://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering
    u = isnothing(u) ? rand() : u #sample if nothing supplied
    N = length(p.w);
    M = (M>0) ? M : N
    Q = cumsum(p.w);
    if !isapprox(Q[N],1)
#        println( "oops degenerate weights: $Q")
        return Particles(p.x,ones(M)/M)
    end
    T = [range(0,stop=1-1/N,length=N) .+ u/N; 1];
    i=1;
    j=1;
    ancestors = zeros(Int, N);
    while (i<=N)
        if (T[i]<Q[j])
            ancestors[i]=j;
            i+=1;
        else
            j+=1;        
        end
    end
    return Particles(p.x[ancestors], ones(M)/M), ancestors
end
