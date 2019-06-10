export computeLikRatio,
       computeCoupledLikRatio,
       runFilter,
       runCoupledFilter

function computeLikRatio(th1::Array{Float}, th2::Array{Float}, observations::Array{Float};
                         u1 = nothing, u2 = nothing, x0::Array = [0, 1.0, 0, 0],
                         N::Int=64, rho::Float=0.95, resampler::Function=resample)
  numRandoms = size(observations,2)*(N+1)
  u1 = (isnothing(u1)) ? randn(numRandoms) : u1
  u2 = (isnothing(u2)) ? randn(numRandoms) : u2
  u3 = rho*u1 + sqrt(1-rho^2)*u2
  ~,lik1 = runFilter(th1, cdf.(Normal(),u1), observations, x0=x0, N=N, resampler=resampler)
  psf2,lik2 = runFilter(th2, cdf.(Normal(),u3), observations, x0=x0, N=N, resampler=resampler)
  diff = lik1 - lik2
  return (psf2,diff)
end

function computeCoupledLikRatio(th1::Array{Float}, th2::Array{Float}, observations::Array{Float};
                         u1 = nothing, u2 = nothing, x0::Array = [0, 1.0, 0, 0],
                         N::Int=64, rho::Float=0.95)
  numRandoms = size(observations,2)*(N+1)
  u1 = (isnothing(u1)) ? randn(numRandoms) : u1
  u2 = (isnothing(u2)) ? randn(numRandoms) : u2
  u3 = rho*u1 + sqrt(1-rho^2)*u2
  psf2,diff = runCoupledFilter(th1, th2, cdf.(Normal(),u1), cdf.(Normal(),u3), observations, x0=x0, N=N)
  return (psf2,diff)
end

function runFilter(theta::Array, u::Union{Array,Nothing}, observations::Array; x0::Array = [0, 1.0, 0, 0],
                   dt::Float=2.0, N::Int=100, filterMethod::String="Aux", resampler::Function=resample)
  @assert length(theta) in [2 3 8]  #TODO: consider better way to provide defaults etc
  if length(theta) == 8
    th = thetaSimple(theta..., dt)
  elseif length(theta) == 2
    th = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, theta..., 0.775, dt)
  elseif length(theta) == 3
    th = thetaSimple(450, theta[1], theta[2], -0.035, 0.015, 0.9, 0.95, theta[3], dt)
  end
  (armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
  hmm = HMM(armondhmmSimple, transll)
  if filterMethod=="Aux"
    prop = auxiliaryprop(armondhmmSimple, x0, approxtrans, approxll)
  elseif filterMethod=="Boot"
    prop = bootstrapprop(armondhmmSimple, x0, transll)
  else
    error("Unknown filter method. Use Aux or Boot instead.")
  end
  (psf, ess, ev) = particlefilter(hmm, observations, N, prop,
                                  resampling=systematicresampling, u=u, resampler=resampler)
  return (psf,ev)
end

function runCoupledFilter(theta1::Array, theta2::Array, u1::Union{Array,Nothing}, u2::Union{Array,Nothing},
                   observations::Array; x0::Array = [0, 1.0, 0, 0],
                   dt::Float=2.0, N::Int=100, filterMethod::String="Aux")
  @assert length(theta1) in [2 3 8]  #TODO: consider better way to provide defaults etc
  if length(theta1) == 8
    th1 = thetaSimple(theta1..., dt)
    th2 = thetaSimple(theta2..., dt)
  elseif length(theta1) == 2
    th1 = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, theta1..., 0.775, dt)
    th2 = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, theta2..., 0.775, dt)
  elseif length(theta1) == 3
    th1 = thetaSimple(450, theta1[1], theta1[2], -0.035, 0.015, 0.9, 0.95, theta1[3], dt)
    th2 = thetaSimple(450, theta2[1], theta2[2], -0.035, 0.015, 0.9, 0.95, theta2[3], dt)
  end
  (armondhmmSimple1, transll1, approxtrans1, approxll1) = armondModelSimple(th1)
  (armondhmmSimple2, transll2, approxtrans2, approxll2) = armondModelSimple(th2)
  hmm1 = HMM(armondhmmSimple1, transll1)
  hmm2 = HMM(armondhmmSimple2, transll2)
  if filterMethod=="Aux"
    prop1 = auxiliaryprop(armondhmmSimple1, x0, approxtrans1, approxll1)
    prop2 = auxiliaryprop(armondhmmSimple2, x0, approxtrans2, approxll2)
  elseif filterMethod=="Boot"
    prop1 = bootstrapprop(armondhmmSimple1, x0, transll1)
    prop2 = bootstrapprop(armondhmmSimple2, x0, transll2)
  else
    error("Unknown filter method. Use Aux or Boot instead.")
  end
  (psf1, psf2, ess, evdiff) = coupledparticlefilter(hmm1, hmm2, observations, N, prop1, prop2,
                                  resampling=systematicresampling, u1=u1, u2=u2, resampler=maxcouplingresample)
  return (psf2, evdiff)
end

