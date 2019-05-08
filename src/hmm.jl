export
    HMM,
    LinearGaussian,
    NonLinearGaussian,
    DiscreteState,
    GaussianHMM,
    generate

abstract type AbstractHMM end
abstract type GaussianHMM end

struct HMM <: AbstractHMM
    transmean::Function
    transloglik::Function # log transition function:  f(xk|xkm1)
    obsmean::Function
    obsloglik::Function   # log observation function: g(y|xk)
    dimx::Int
    dimy::Int
end

struct LinearGaussian <: GaussianHMM
    #=
        x <- Ax + cholesky(Q)' * randn
        y <- Bx + cholesky(R)' * randn
    =#
    A::Matrix{Float}
    B::Matrix{Float}
    Q::Union{Float,Matrix{Float}}
    R::Union{Float,Matrix{Float}}
    # implicit
    dimx::Int
    dimy::Int
    cholQ::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    cholR::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    transmean::Function
    obsmean::Function
    function LinearGaussian(A,B,Q,R)
        dimx=size(A,1)
        dimy=size(B,1)
        @assert dimx==size(A,2)==size(Q,1)==size(Q,2) "dimensions don't match"
        @assert dimy==size(R,1)==size(R,2) "dimensions don't match"
        @assert issymmetric(Q) && issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(Q) && isposdef(R) "cov mat must be pos def"
        new(A,B,Q,R,dimx,dimy,cholesky(Q).U,cholesky(R).U,
            (k,xkm1) -> A*xkm1,
            (k,xk)   -> B*xk )
    end
end

struct NonLinearGaussian <: GaussianHMM
    #=
        x <- f(x) + cholesky(Q)' * randn
        y <- g(x) + cholesky(R)' * randn

        No longer assume gaussian noise on all state variables
    =#
    transmean::Function
    obsmean::Function
    Q::Union{Float,Matrix{Float}}
    R::Union{Float,Matrix{Float}}
    indQNoise::Array
    indRNoise::Array
    dimx::Int
    dimy::Int
    # implicit
    cholQ::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    cholR::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    function NonLinearGaussian(f, g, Q, R, indQNoise, indRNoise, dimx, dimy)
        @assert size(Q,1) == length(indQNoise) "must specify cov for noisy vars" 
        @assert size(R,1) == length(indRNoise) "must specify cov for noisy obs"
        @assert issymmetric(Q) && issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(Q) && isposdef(R) "cov mat must be pos def"
        new((k,xkm1)-> f(k,xkm1),
            (k,xk)  -> g(k,xk),
            Q, R, indQNoise, indRNoise,
            dimx, dimy, cholesky(Q).U, cholesky(R).U)
    end
end

struct DiscreteState <: GaussianHMM
    #=
        x <- f(x) via Markov jump process
        y <- g(x,y) + cholesky(R)' * randn for observations
        Designed for armond model to formulate without observation noise
    =#
    transmean::Function
    obsmean::Function
    R::Union{Float,Matrix{Float}}
    dimx::Int
    dimy::Int
    # implicit
    cholR::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    function DiscreteState(f, g, R, dimx, dimy)
        @assert issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(R) "cov mat must be pos def"
        new((k, xkm1)-> f(k, xkm1),
            (k, xk, ykm1)  -> g(k, xk, ykm1),
            R, dimx, dimy, cholesky(R).U)
    end
end


function HMM(g::LinearGaussian)
    transloglik = (k, xkm1, xk) -> -norm(g.cholQ'\(xk - g.transmean(k,xkm1)))^2/2
    obsloglik   = (k, ykm1, yk, xk) -> -norm(g.cholR'\(yk - g.obsmean(k,xk)))^2/2
    HMM(g.transmean, transloglik, g.obsmean, obsloglik, g.dimx, g.dimy)
end

function HMM(g::NonLinearGaussian, transloglik)
    #set default transloglik as for linear gaussian case, but allow user to specify in more complicated cases
    transloglik = (isnothing(transloglik)) ? (k,xkm1,xk) -> -norm(g.cholQ'\(xk - g.transmean(k,xkm1)))^2/2 : transloglik
    obsloglik   = (k, ykm1, yk, xk) -> -norm(g.cholR'\(yk - g.obsmean(k,xk)))^2/2
    HMM(g.transmean, transloglik, g.obsmean, obsloglik, g.dimx, g.dimy)
end

function HMM(g::DiscreteState, transloglik)
    obsloglik   = (k, ykm1, yk, xk) -> -norm(g.cholR'\(yk - g.obsmean(k,xk,ykm1)))^2/2
    HMM(g.transmean, transloglik, g.obsmean, obsloglik, g.dimx, g.dimy)
end

### Generation of observations

"""
    generate(lg, x0, K)

Generate observations following a given (non)linear Gaussian dynamic for `K`
time steps.
"""
function generate(g::DiscreteState, x0::Vector{Float}, y0::Vector{Float}, K::Int
                    )::Tuple{Matrix{Float},Matrix{Float}}
    @assert length(x0)==g.dimx "dimensions don't match"
    @assert length(y0)==g.dimy "dimensions don't match"
    # allocate states/observations
    states, observations = zeros(g.dimx, K), zeros(g.dimy, K)
    # assign first state
    states[:,1] = x0
    observations[:,1] = y0
    states[:,2] = g.transmean(2,states[:,1])
println(states)
    noisey = g.cholR' * randn(g.dimy,K)
    # use noise in iterative linear system
    for k = 2:(K-1)
println(k, " of ", K)
        observations[:,k] = g.obsmean(k, states[:,k], observations[:,k-1]) + noisey[:,k]
println(observations[:,k])
        states[:,k+1]     = g.transmean(k+1, states[:,k])
    end
    # last observation
    observations[:,K] = g.obsmean(K, states[:,K], observations[:,K-1]) + noisey[:,K]
    # package and return
    return (states, observations)
end

function generate(g::NonLinearGaussian, x0::Vector{Float}, K::Int
                    )::Tuple{Matrix{Float},Matrix{Float}}
    @assert length(x0)==g.dimx "dimensions don't match"
    # allocate states/observations
    states, observations = zeros(g.dimx, K), zeros(g.dimy, K)
    # assign first state
    states[:,1] = x0
    # pre-generate noise
    noiseDimX = length(g.indQNoise)
    noiseDimY = length(g.indRNoise)
    noisex = zeros(g.dimx,K)
    noisey = zeros(g.dimy,K)
    noisex[g.indQNoise,:] = g.cholQ' * randn(noiseDimX,K)
    noisey[g.indRNoise,:] = g.cholR' * randn(noiseDimY,K)
    # use noise in iterative linear system
    for k = 1:(K-1)
        observations[:,k] = g.obsmean(k, states[:,k]) + noisey[:,k]
        states[:,k+1]     = g.transmean(k+1, states[:,k]) + noisex[:,k+1]
    end
    # last observation
    observations[:,K] = g.obsmean(K, states[:,K]) + noisey[:,K]
    # package and return
    return (states, observations)
end

"""
    generate(lg, x0, K)
Generate observations following a given (non)linear Gaussian dynamic for `K`
time steps.
"""
function generate(g::LinearGaussian, x0::Vector{Float}, K::Int
                    )::Tuple{Matrix{Float},Matrix{Float}}
    @assert length(x0)==g.dimx "dimensions don't match"
    # allocate states/observations
    states, observations = zeros(g.dimx, K), zeros(g.dimy, K)
    # assign first state
    states[:,1] = x0
    # pre-generate noise
    noisex = g.cholQ' * randn(g.dimx,K)
    noisey = g.cholR' * randn(g.dimy,K)
    # use noise in iterative linear system
    for k = 1:(K-1)
        observations[:,k] = g.obsmean(k, states[:,k]) + noisey[:,k]
        states[:,k+1]     = g.transmean(k+1, states[:,k]) + noisex[:,k+1]
    end
    # last observation
    observations[:,K] = g.obsmean(K, states[:,K]) + noisey[:,K]
    # package and return
    return (states, observations)
end
