export
    Proposal,
    bootstrapprop,
    auxiliaryprop

struct Proposal
    mu0::Union{Float,Vector{Float}}
    noise::Function
    mean::Function
    loglik::Function
end

function bootstrapprop(g::LinearGaussian, mu0::Union{Float,Vector{Float}}=0.0)
    hmm = HMM(g)
    n = nothing
    Proposal(
        (mu0==0.0) ? (g.dimx>1 ? zeros(g.dimx) : mu0 ) : mu0,
        (g.dimx>1) ? (k=n)->(g.cholQ'*randn(g.dimx)) : ((k=n)->g.cholQ'*randn()),
        (k=n,xkm1=n,yk=n)      -> hmm.transmean(k, xkm1),
        (k=n,xkm1=n,yk=n,xk=n) -> hmm.transloglik(k, xkm1, xk)
    )
end

function bootstrapprop(g::NonLinearGaussian, mu0::Union{Float,Vector{Float}}=0.0, transloglik::Function=nothing)
#println(mu0)
    hmm = HMM(g, transloglik)
    n = nothing
    function noiseFun(k=n)
    noiseDimX = length(g.indQNoise)
    noisex = zeros(g.dimx)
    noisex[g.indQNoise] = g.cholQ' * randn(noiseDimX)
    return noisex
    end
    Proposal(
        (mu0==0.0) ? (g.dimx>1 ? zeros(g.dimx) : mu0 ) : mu0,
        (k=n)-> noiseFun(k),
        (k=n,xkm1=n,yk=n)      -> hmm.transmean(k, xkm1),
        (k=n,xkm1=n,yk=n,xk=n) -> hmm.transloglik(k, xkm1, xk)
    )
end

function auxiliaryprop(g::NonLinearGaussian, mu0::Union{Float,Vector{Float}}=0.0,
                       approxtransmean::Function=nothing, approxloglik::Function=nothing)
    n = nothing
    function noiseFun(k=n)
    noiseDimX = length(g.indQNoise)
    noisex = zeros(g.dimx)
    noisex[g.indQNoise] = g.cholQ' * randn(noiseDimX)
    return noisex
    end
    Proposal(
        (mu0==0.0) ? (g.dimx>1 ? zeros(g.dimx) : mu0 ) : mu0,
        (k=n)-> noiseFun(k),
        (k=n,xkm1=n,yk=n)      -> approxtransmean(k,xkm1,yk),
        (k=n,xkm1=n,yk=n,xk=n) -> approxloglik(k,xkm1, yk, xk)
    )
end

# # reverese bootstrap (reversed dynamic)
# function rbootstrap(lg::LinearGaussian)
#     hmm = HMM(lg)
#     n   = nothing
#     pia = pinv(lg.A)
#     Proposal(
#         (k=n)                  -> pia * lg.cholQ' * randn(lg.dimx)
#         (k=n,xkp1=n,yk=n)      -> pia * xkp1
#         (k=n,xk=n,yk=n,xkp1)   -> hmm.transloglik(k, xk, xkp1)
#     )
# end
