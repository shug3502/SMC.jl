using Distributions

export
    correlated

#use correlated pseudo marginal method to gain efficiency compared to pmcmc
#See deligiannis et al 2018 and golightly et al 2018

function runFilter(theta::Array, u::Array, observations::Array; x0::Array = [0, 1.0, 0, 0], N::Int=100)
  @assert length(theta)==2 || length(theta)==8  #TODO: consider better way to provide defaults etc
  if length(theta) == 2
    th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, theta[1], theta[2], 0.775, 3.4)
  else
    th = thetaSimple(theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], 3.4)
  end
  (armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
  hmm = HMM(armondhmmSimple, transll)
  auxprop = auxiliaryprop(armondhmmSimple, x0, approxtrans, approxll)
  (psf, ess, ev) = particlefilter(hmm, observations, N, auxprop)
  return ev
end

function correlated(observations::Array, priors::Array,
         paramProposal::Array, dimParams::Int, numRandoms::Int;
         rho::Float=0.9, numIter::Int=1000, N::Int=100)
  #priors should be an array of distributions

  @assert length(priors) == dimParams
  @assert length(paramProposal) == dimParams  

  acceptances = 0
  c = zeros(dimParams,numIter)
  #i=1 case; set c(1) in the support of prior and draw random numbers
  for j=1:dimParams
    c[j,1] = rand(priors[j])
  end
  u = randn(numRandoms)
  lik = runFilter(c[:,1],u,observations,N=N)

  #iterate
  for i=2:numIter
    if i%100 == 0
      println("Iter: ", i)
      println(theta)
      println("acceptance rate is: ", acceptances/i)
    end
    #propose new params
    cPrime = [rand(paramProposal[j](c[j,i-1])) for j in 1:dimParams]
    w = randn(numRandoms)
    uPrime = rho*u + sqrt(1-rho^2)*w
    #TODO: needs to take in random numbers appropriately
    likPrime = runFilter(cPrime,uPrime,observations,N=N)
    acceptanceProb = sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) - 
                     sum([logpdf(priors[j],c[j,i-1]) for j in 1:dimParams]) + 
                     sum([logpdf(paramProposal[j](c[j,i-1]),c[j,i-1]) for j in 1:dimParams]) - 
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
  return (c, acceptances/numIter)
end
