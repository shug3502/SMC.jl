using Distributions, Distributed

export
    correlated,
    runFilter,
    runCoupledFilter

#use correlated pseudo marginal method to gain efficiency compared to pmcmc
#See deligiannis et al 2018 and golightly et al 2018

function runFilter(theta::Array, u::Union{Array,Nothing}, observations::Array; x0::Array = [0, 1.0, 0, 0],
                   dt::Float=2.0, N::Int=100, filterMethod::String="Aux", resampler::Function=resample)
  @assert length(theta)==2 || length(theta)==8  #TODO: consider better way to provide defaults etc
  if length(theta) == 2
    th = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, theta[1], theta[2], 0.775, dt)
  else
    th = thetaSimple(theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], dt)
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
  return ev
end

function runCoupledFilter(theta1::Array, theta2::Array, u1::Union{Array,Nothing}, u2::Union{Array,Nothing},
                   observations::Array; x0::Array = [0, 1.0, 0, 0],
                   dt::Float=2.0, N::Int=100, filterMethod::String="Aux")
  @assert length(theta1)==2 || length(theta1)==8  #TODO: consider better way to provide defaults etc
  if length(theta1) == 2
    th1 = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, theta1[1], theta1[2], 0.775, dt)
    th2 = thetaSimple(450, 0.008, 0.025, -0.035, 0.015, theta2[1], theta2[2], 0.775, dt)
  else
    th1 = thetaSimple(theta1[1], theta1[2], theta1[3], theta1[4], theta1[5], theta1[6], theta1[7], theta1[8], dt)
    th2 = thetaSimple(theta2[1], theta2[2], theta2[3], theta2[4], theta2[5], theta2[6], theta2[7], theta2[8], dt)
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
  return evdiff
end


function correlated(observations::Array, priors::Array,
         paramProposal::Array, dimParams::Int, numRandoms::Int;
         rho::Float=0.9, numIter::Int=1000, N::Int=100,
         initialisationFn=nothing, printFreq::Int=1000,
         resampler::Function=resample)
  #priors should be an array of distributions

  @assert length(priors) == dimParams
  @assert length(paramProposal) == dimParams  

  c = zeros(dimParams,numIter)
  #i=1 case; 
  if isnothing(initialisationFn)
    #By default, set c(1) in the support of prior
    for j=1:dimParams
      c[j,1] = rand(priors[j])
    end
  else
    #for very broad priors a more informative initialisation can be required
    for j=1:dimParams
      c[j,1] = rand(initialisationFn[j])
    end
  end
  acceptances = 0
  u = randn(numRandoms) #draw some random numbers to pass to filter calc
  uUnif = cdf.(Normal(),u)
  lik = runFilter(c[:,1],uUnif,observations,N=N,resampler=resampler)

  #iterate
  for i=2:numIter
    if i%printFreq == 0
      println("Iter: ", i)
      println("acceptance rate is: ", acceptances/i)
    end
    #propose new params
    cPrime = [rand(paramProposal[j](c[j,i-1])) for j in 1:dimParams]
    w = randn(numRandoms)
    uPrime = rho*u + sqrt(1-rho^2)*w
    uPrimeUnif = cdf.(Normal(),uPrime) #convert from gaussian to uniform
    likPrime = runFilter(cPrime,uPrimeUnif,observations,N=N,resampler=resampler)
    acceptanceProb = sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) - 
                     sum([logpdf(priors[j],c[j,i-1]) for j in 1:dimParams]) + 
                     sum([logpdf(paramProposal[j](cPrime[j]),c[j,i-1]) for j in 1:dimParams]) - 
                     sum([logpdf(paramProposal[j](c[j,i-1]),cPrime[j]) for j in 1:dimParams]) + 
                     + likPrime - lik
    if log(rand()) < acceptanceProb 
      #then accept
      c[:,i] = cPrime
      u = deepcopy(uPrime) #note not storing the u vars
      lik = deepcopy(likPrime)      
      acceptances += 1
      else 
      c[:,i] = c[:,i-1]
    end
  end
  return (transpose(c), acceptances/numIter)
end

