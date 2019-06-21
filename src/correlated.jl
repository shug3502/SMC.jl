using Distributions, Distributed
using Statistics: cov

export
    correlated,
    noisyMCMC

#use correlated pseudo marginal method to gain efficiency compared to pmcmc
#See deligiannis et al 2018 and golightly et al 2018

function correlated(observations::Array, priors::Array,
         paramProposal::Function, dimParams::Int, numRandoms::Int;
         rho::Float=0.95, numIter::Int=1000, x0::Array=[0,1,0,0], N::Int=100,
         dt::Float=2.0, initialisationFn=nothing, printFreq::Int=1000,
         model::String="Simple",
         resampler::Function=resample,
         burnin::Int=round(Int,numIter/10))
  #priors should be an array of distributions

  @assert length(priors) == dimParams

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
  X_1toK, lik = runFilter(c[:,1],uUnif,observations,x0=x0,N=N,dt=dt,model=model,resampler=resampler)
  hiddenstates[:,1] = mapslices(x -> findfirst(w -> w>0, x),X_1toK,dims=1)

  #iterate
  for i=2:numIter
    if i%printFreq == 0
      println("Iter: ", i)
      println("acceptance rate is: ", acceptances/i)
    end
    #propose new params
    cPrime = rand(paramProposal(c[:,i-1]))
    if sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) > -Inf
      #then is in support of the prior and can continue
      w = randn(numRandoms)
      uPrime = rho*u + sqrt(1-rho^2)*w
      uPrimeUnif = cdf.(Normal(),uPrime) #convert from gaussian to uniform
      X_1toK, likPrime = runFilter(cPrime,uPrimeUnif,observations,x0=x0,N=N,dt=dt,model=model,resampler=resampler)
      acceptanceProb = sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) - 
                     sum([logpdf(priors[j],c[j,i-1]) for j in 1:dimParams]) + 
                     logpdf(paramProposal(cPrime),c[:,i-1]) - 
                     logpdf(paramProposal(c[:,i-1]),cPrime) + 
                     likPrime - lik
    else #not in support of prior
      acceptanceProb = -Inf
    end

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
    if (i==burnin)
      if (acceptances > 0.001*burnin)
      println("Burn in completed. Setting optimized proposal and beginning sampling...")
      #adapt the parameter proposal distribution
      covPilot = cov(transpose(c[:,1:burnin]))
      scalingFactor = 2.56^2/dimParams #see Golightly et al 2017, Sherlock et al 2015
      #set optimized Proposal
      Sigma = nearestSPD(covPilot*scalingFactor) #ensure positive definite
      println(diag(Sigma))
      paramProposal = x -> MvNormal(x,Sigma);
      else
      @warn "Insufficient acceptances to optimize proposal properly. Skipping adaption."
      end
    end
  end
  return (transpose(c[:,(burnin+1):numIter]), transpose(hiddenstates[:,(burnin+1):numIter]), acceptances/numIter)
end

function noisyMCMC(observations::Array, priors::Array,
         paramProposal::Function, dimParams::Int, numRandoms::Int;
         rho::Float=0.95, numIter::Int=1000, x0::Array=[0,1,0,0], N::Int=100,
         dt::Float=2.0, initialisationFn=nothing, printFreq::Int=1000,
         model::String="Simple",
         resampler::Function=resample,
         burnin::Int=round(Int,numIter/10))
  #priors should be an array of distributions

  @assert length(priors) == dimParams

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
  X_1toK, lik = runFilter(c[:,1],uUnif,observations,x0=x0,N=N,dt=dt,model=model,resampler=resampler)
  hiddenstates[:,1] = mapslices(x -> findfirst(w -> w>0, x),X_1toK,dims=1)

  #iterate
  for i=2:numIter
    if i%printFreq == 0
      println("Iter: ", i)
      println("acceptance rate is: ", acceptances/i)
    end
    #propose new params
    cPrime = rand(paramProposal(c[:,i-1]))
    if sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) > -Inf #params are in support of prior 
      w = randn(numRandoms)
      X_1toK, likDiff = computeCoupledLikRatio(c[:,i-1], cPrime, observations,
                         u1 = u, u2 = w, x0 = x0, dt = dt,
                         N=N, rho=rho, model=model)
      acceptanceProb = sum([logpdf(priors[j],cPrime[j]) for j in 1:dimParams]) - 
                       sum([logpdf(priors[j],c[j,i-1]) for j in 1:dimParams]) + 
                       logpdf(paramProposal(cPrime),c[:,i-1]) - 
                       logpdf(paramProposal(c[:,i-1]),cPrime) + 
                       - likDiff
    else
      acceptanceProb = -Inf  #params not in support of prior
    end
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
    if i==burnin
      println("Burn in completed. Setting optimized proposal and beginning sampling...")
      #adapt the parameter proposal distribution
      covPilot = cov(transpose(c[:,1:burnin]))
      scalingFactor = 2.56^2/dimParams #see Golightly et al 2017, Sherlock et al 2015
      #set optimized Proposal
      Sigma = nearestSPD(covPilot*scalingFactor) #ensure positive definite
      println(diag(Sigma))
      paramProposal = x -> MvNormal(x,Sigma);
    end
  end
  return (transpose(c[:,(burnin+1):numIter]), transpose(hiddenstates[:,(burnin+1):numIter]), acceptances/numIter)
end

