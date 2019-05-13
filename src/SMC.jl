module SMC

using Compat
using Distributions
using LinearAlgebra
using Statistics
using Distributed
import Base.length

const Int   = Int64
const Float = Float64

include("hmm.jl")
include("kalman.jl")
include("helpers.jl")
include("particles.jl")
include("resample.jl")
include("proposal.jl")
include("particlefilter.jl")
include("particlesmoother.jl")
include("armondmodel.jl")
include("armondmodelSimple.jl")
#include("smc2.jl")
include("correlated.jl")

end # module
