export computeLikRatio,
       computeCoupledLikRatio,
       runFilter,
       runCoupledFilter,
       constructProposal

#TODO: put everything in an options structure?
function computeLikRatio(th1::Array{Float}, th2::Array{Float}, observations::Array{Float};
                         u1 = nothing, u2 = nothing, x0::Array = [0, 1.0, 0, 0], dt::Float=2.0,
                         N::Int=64, rho::Float=0.95, model::String="Simple", resampler::Function=resample)
  numRandoms = size(observations,2)*(N+1)
  u1 = (isnothing(u1)) ? randn(numRandoms) : u1
  u2 = (isnothing(u2)) ? randn(numRandoms) : u2
  u3 = rho*u1 + sqrt(1-rho^2)*u2
  ~,lik1 = runFilter(th1, cdf.(Normal(),u1), observations, x0=x0, N=N, dt=dt, model=model, resampler=resampler)
  X_1toK,lik2 = runFilter(th2, cdf.(Normal(),u3), observations, x0=x0, N=N, dt=dt, model=model, resampler=resampler)
  diff = lik1 - lik2
  return (X_1toK,diff)
end

function computeCoupledLikRatio(th1::Array{Float}, th2::Array{Float}, observations::Array{Float};
                         u1 = nothing, u2 = nothing, x0::Array = [0, 1.0, 0, 0], dt::Float=2.0,
                         N::Int=64, rho::Float=0.95, model::String="Simple")
  numRandoms = size(observations,2)*(N+1)
  u1 = (isnothing(u1)) ? randn(numRandoms) : u1
  u2 = (isnothing(u2)) ? randn(numRandoms) : u2
  u3 = rho*u1 + sqrt(1-rho^2)*u2
  X_1toK,diff = runCoupledFilter(th1, th2, cdf.(Normal(),u1), cdf.(Normal(),u3), observations, x0=x0, N=N, dt=dt, model=model)
  return (X_1toK,diff)
end

function runFilter(theta::Array, u::Union{Array,Nothing}, observations::Array; x0::Array = [0, 1.0, 0, 0],
                   dt::Float=2.0, N::Int=100, filterMethod::String="Aux", model::String="Simple", resampler::Function=resample)
  th, prop, hmm = constructProposal(theta, model, dt)
  (psf, ancestors, ess, ev) = particlefilter(hmm, observations, N, prop,
                                  resampling=systematicresampling, u=u, resampler=resampler)
  #psw = particlesmoother_ffbs(hmm, psf)
  psw = deepcopy(psf)
  K = length(psw)
  X_1toK = zeros(hmm.dimx,K)
  for k=1:K
    X_1toK[:,k] = psw.p[k].x[rand(Categorical(psw.p[k].w))]
  end
  return (X_1toK, ev)
end

function runCoupledFilter(theta1::Array, theta2::Array, u1::Union{Array,Nothing}, u2::Union{Array,Nothing},
                   observations::Array; x0::Array = [0, 1.0, 0, 0],
                   dt::Float=2.0, N::Int=100, filterMethod::String="Aux", model::String="Simple")
#=
  if model == "Simple"
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
    (hmmModel1, transll1, approxtrans1, approxll1) = armondModelSimple(th1)
    (hmmModel2, transll2, approxtrans2, approxll2) = armondModelSimple(th2)
  elseif model=="Anaphase"
    th1 = thetaAnaphase(theta1..., dt)
    th2 = thetaAnaphase(theta2..., dt)
    (hmmModel1, transll, approxtrans, approxll) = anaphaseModel(th)
    (hmmModel2, transll, approxtrans, approxll) = anaphaseModel(th)
  else
    println("Not yet implemented")
    @assert false
  end
  hmm1 = HMM(hmmModel1, transll1)
  hmm2 = HMM(hmmModel2, transll2)
  if filterMethod=="Aux"
    prop1 = auxiliaryprop(hmmModel1, x0, approxtrans1, approxll1)
    prop2 = auxiliaryprop(hmmModel2, x0, approxtrans2, approxll2)
  elseif filterMethod=="Boot"
    prop1 = bootstrapprop(hmmModel1, x0, transll1)
    prop2 = bootstrapprop(hmmModel2, x0, transll2)
  else
    error("Unknown filter method. Use Aux or Boot instead.")
  end
=#
th1, prop1, hmm1 = constructProposal(theta1, model, dt)
th2, prop2, hmm2 = constructProposal(theta2, model, dt)
  (psf1, psf2, ancestors, ess, evdiff) = coupledparticlefilter(hmm1, hmm2, observations, N, prop1, prop2,
                                  resampling=systematicresampling, u1=u1, u2=u2, model=model, resampler=maxcouplingresample)
#  psw = particlesmoother_ffbs(hmm2, psf2)
  psw = deepcopy(psf2)
  K = length(psw)
  X_1toK = zeros(hmm2.dimx,K)
  for k=1:K          
    X_1toK[:,k] = psw.p[k].x[rand(Categorical(psw.p[k].w))]
  end

  return (X_1toK, evdiff)
end

function constructProposal(theta::Array, model::String, dt::Float)
  if model == "Simple"
    @assert length(theta) in [2 3 8]  #TODO: consider better way to provide defaults etc
    if length(theta) == 8
      th = thetaSimple(theta..., dt)
    elseif length(theta) == 2
      th = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, theta..., 0.775, dt)
    elseif length(theta) == 3
      th = thetaSimple(450, theta[1], theta[2], -0.035, 0.015, 0.9, 0.95, theta[3], dt)
    end
    (hmmModel, transll, approxtrans, approxll) = armondModelSimple(th)
  elseif model == "Anaphase"
    th = thetaAnaphase(theta..., dt)
    (hmmModel, transll, approxtrans, approxll) = anaphaseModel(th)
  elseif model == "TensionClock"
    println("Not yet implemented")
    @assert false
  else
    println("Not yet implemented")
    @assert false
  end
  hmm = HMM(hmmModel, transll)
  if filterMethod=="Aux"
    prop = auxiliaryprop(hmmModel, x0, approxtrans, approxll)
  elseif filterMethod=="Boot"
    prop = bootstrapprop(hmmModel, x0, transll)
  else
    error("Unknown filter method. Use Aux or Boot instead.")
  end
return (th, prop, hmm)
end

