export
    Particles,
    ParticleSet,
    length,
    mean,
    evidence

const ParticleType = Union{Float, Vector{Float}}

mutable struct Particles{T <: ParticleType}
    x::Vector{T}     # N x (1) or N x (dimx)
    w::Vector{Float} # N
end

mutable struct ParticleSet{T <: ParticleType}
    p::Vector{Particles{T}} # T
end

mutable struct ParticleSetTheta{T <: ParticleType}
    q::Vector{Particles{T}} #Nthet
end

"""
    Particles(N,dx)

Create a Particles object with `N` particles each of dimension `dx`.
"""
Particles(N::Int, dx::Int=1) =
    Particles( dx==1 ? zeros(N) : Vector{Vector{Float}}(), ones(N)/N )

"""
    ParticleSet(N,dx,K)

Create a set of `K` Particles with `N` particles of dimension `dx`. This is for
a HMM with `K` steps.
"""
ParticleSet(N::Int, dx::Int, K::Int) =
    ParticleSet( [Particles(N,dx) for i in 1:K] )

"""
    length(p::Particles)

Number of particles.
"""
length(p::Particles)    = length(p.w)

"""
    length(ps::ParticleSet)

Number of slices (steps in the HMM).
"""
length(ps::ParticleSet) = length(ps.p)

"""
    mean(p::Particles)

Compute the mean corresponding to the particles `p`.
"""
mean(p::Particles) = sum(p.x[i] * p.w[i] for i = 1:length(p))

"""
    mean(ps::ParticleSet)

Compute the mean corresponding to a particle set
"""
mean(ps::ParticleSet) = [mean(p) for p in ps.p]

"""
    evidence(p::Particles)

Compute the likelihood p(y | theta) marginalizing over hidden states
"""
evidence(p::Particles) = -length(p) + log(sum(p.w[i] for i = 1:length(p)))

"""
    evidence(ps::ParticleSet)

Compute the likelihood p(y | theta) of a particle set
"""
evidence(ps::ParticleSet) = sum([evidence(p) for p in ps.p])

#=
"""
  ParticleSetTheta(Ntheta,Nx,dtheta,dx,K)
Particles in X and Theta
"""
ParticleSetTheta(Ntheta::Int,Nx::Int,dtheta::Int,dx::Int,K::Int) = 
    ParticleSetTheta( [[Particles(Nx,dx) for i in 1:K] for j in 1:Ntheta] )
=#
