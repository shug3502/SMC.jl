using Distributions, Distributed

export
    correlated,
    noisyMCMC

#use correlated pseudo marginal method to gain efficiency compared to pmcmc
#See deligiannis et al 2018 and golightly et al 2018

function correlated(observations::Array, priors::Array,
         paramProposal::Array, dimParams::Int, numRandoms::Int;
         rho::Float=0.95, numIter::Int=1000, N::Int=100,
         initialisationFn=nothing, printFreq::Int=1000,
         resampler::Function=resample)
  #priors should be an array of distributions

  @assert length(priors) == dimParams
  @assert length(paramProposal) == dimParams  

  c = zeros(dimParams,numIter)
  hiddenstates = zeros(Int, size(observations,2), numIter) #binary states converted to index
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
  X_1toK, lik = runFilter(c[:,1],uUnif,observations,N=N,resampler=resampler)
  hiddenstates[:,1] = mapslices(x -> findfirst(w -> w>0, x),X_1toK,dims=1)

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
    X_1toK, likPrime = runFilter(cPrime,uPrimeUnif,observations,N=N,resampler=resampler)
    acceptanceProb = sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) - 
                     sum([logpdf(priors[j],c[j,i-1]) for j in 1:dimParams]) + 
                     sum([logpdf(paramProposal[j](cPrime[j]),c[j,i-1]) for j in 1:dimParams]) - 
                     sum([logpdf(paramProposal[j](c[j,i-1]),cPrime[j]) for j in 1:dimParams]) + 
                     + likPrime - lik
    if log(rand()) < acceptanceProb 
      #then accept
      c[:,i] = cPrime
      hiddenstates[:,i] = mapslices(x -> findfirst(w -> w>0, x),X_1toK,dims=1)
      u = deepcopy(uPrime) #note not storing the u vars
      lik = deepcopy(likPrime)      
      acceptances += 1
      else 
      c[:,i] = c[:,i-1]
      hiddenstates[:,i] = hiddenstates[:,i-1]
    end
  end
  return (transpose(c), transpose(hiddenstates), acceptances/numIter)
end

function noisyMCMC(observations::Array, priors::Array,
         paramProposal::Array, dimParams::Int, numRandoms::Int;
         rho::Float=0.95, numIter::Int=1000, N::Int=100,
         initialisationFn=nothing, printFreq::Int=1000,
         resampler::Function=resample)
  #priors should be an array of distributions

  @assert length(priors) == dimParams
  @assert length(paramProposal) == dimParams  

  c = zeros(dimParams,numIter)
  hiddenstates = zeros(Int, size(observations,2), numIter) #binary states converted to index
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
  X_1toK, lik = runFilter(c[:,1],uUnif,observations,N=N,resampler=resampler)
  hiddenstates[:,1] = mapslices(x -> findfirst(w -> w>0, x),X_1toK,dims=1)

  #iterate
  for i=2:numIter
    if i%printFreq == 0
      println("Iter: ", i)
      println("acceptance rate is: ", acceptances/i)
    end
    #propose new params
    cPrime = [rand(paramProposal[j](c[j,i-1])) for j in 1:dimParams]
    w = randn(numRandoms)
    X_1toK, likDiff = computeCoupledLikRatio(c[:,i-1], cPrime, observations,
                         u1 = u, u2 = w,
                         N=N, rho=rho)
    acceptanceProb = sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) - 
                     sum([logpdf(priors[j],c[j,i-1]) for j in 1:dimParams]) + 
                     sum([logpdf(paramProposal[j](cPrime[j]),c[j,i-1]) for j in 1:dimParams]) - 
                     sum([logpdf(paramProposal[j](c[j,i-1]),cPrime[j]) for j in 1:dimParams]) + 
                     - likDiff
    if log(rand()) < acceptanceProb 
      #then accept
      c[:,i] = cPrime
      hiddenstates[:,i] = mapslices(x -> findfirst(w -> w>0, x),X_1toK,dims=1)
      u = rho*u + sqrt(1-rho^2)*w #note not storing the u vars
      acceptances += 1
      else 
      c[:,i] = c[:,i-1]
      hiddenstates[:,i] = hiddenstates[:,i-1]
    end
  end
  return (transpose(c), transpose(hiddenstates), acceptances/numIter)
end

