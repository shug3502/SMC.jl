export
    HMM,
    LinearGaussian,
    generate

abstract type AbstractHMM end

struct HMM <: AbstractHMM
    transmean::Function
    transloglik::Function # log transition function:  f(xk|xkm1)
    obsmean::Function
    obsloglik::Function   # log observation function: g(y|xk)
    dimx::Int
    dimy::Int
end

struct LinearGaussian <: AbstractHMM
    #=
        x <- Ax + chol(Q)' * randn
        y <- Bx + chol(Q)' * randn
    =#
    A::Matrix{Float}
    B::Matrix{Float}
    Q::Matrix{Float}
    R::Matrix{Float}
    # implicit
    dimx::Int
    dimy::Int
    cholQ::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    cholR::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    hmm::HMM
    function LinearGaussian(A,B,Q,R)
        dimx=size(A,1)
        dimy=size(B,1)
        @assert dimx==size(A,2)==size(Q,1)==size(Q,2) "dimensions don't match"
        @assert dimy==size(R,1)==size(R,2) "dimensions don't match"
        @assert issymmetric(Q) && issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(Q) && isposdef(R) "cov mat must be pos def"
        new(A,B,Q,R,dimx,dimy,chol(Q),chol(R))
    end
end

function HMM(lg::LinearGaussian)
    # means
    transmean   = (k,xkm1) -> lg.A*xkm1
    obsmean     = (k,xk)   -> lg.B*xk
    # loglikelihoods
    transloglik = (k,xkm1,xk) -> -norm(lg.cholQ'\(xk - transmean(k,xkm1)))^2/2
    obsloglik   = (k,xk,  yk) -> -norm(lg.cholR'\(yk - obsmean(k,xk)))^2/2
    # Package and return
    HMM(transmean, transloglik, obsmean, obsloglik, lg.dimx, lg.dimy)
end

### Generation of observations

"""
    generate(lg, x0, T)

Generate observations following a given dynamic for `T` time steps.
"""
function generate(lg::LinearGaussian, x0::Vector{Float}, K::Int
                    )::Tuple{Matrix{Float},Matrix{Float}}
    @assert length(x0)==lg.dimx "dimensions don't match"
    # allocate states/observations
    states, observations = zeros(lg.dimx, K), zeros(lg.dimy, K)
    # assign first state
    states[:,1] = x0
    # pre-generate noise
    noisex = lg.cholQ' * randn(lg.dimx,K)
    noisey = lg.cholR' * randn(lg.dimy,K)
    # use noise in iterative linear system
    for k = 1:(K-1)
        observations[:,k] = lg.B*states[:,k] + noisey[:,k]
        states[:,k+1]     = lg.A*states[:,k] + noisex[:,k+1]
    end
    # last observation
    observations[:,K] = lg.B*states[:,K] + noisey[:,K]
    # package and return
    return (states, observations)
end
