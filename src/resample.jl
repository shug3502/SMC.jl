export
    resample,
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
                  )::Tuple{Particles,Float}
    ess = 1.0/sum(p.w.^2)
    N   = length(p)
    M   = M>0 ? M : N
    (M != N || ess < essthresh * N) ? (rs(p, M, u), ess) : (p, ess)
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
    mask = [j for i in 1:N for j in ones(Int,ni[i])*i]
    Particles(p.x[mask], ones(M)/M)
end

"""
    systematicresampling(p::Particles, M, u)
Systematic resampling of a particles opject `p`.
Helpful when wanting to correlate randomness in successive filter evaluations.
"""
function systematicresampling(p::Particles, M::Int=0, u=nothing)::Particles
    #based on MATLAB code from https://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering
    u = isnothing(u) ? rand() : u #sample if nothing supplied
    N = length(p.w);
    M = (M>0) ? M : N
    Q = cumsum(p.w);
    T = [range(0,stop=1-1/N,length=N) .+ u/N; 1];
    i=1;
    j=1;
    indx = zeros(Int, N);
    while (i<=N)
        if (T[i]<Q[j])
            indx[i]=j;
            i+=1;
        else
            j+=1;        
        end
    end
    Particles(p.x[indx], ones(M)/M)
end
