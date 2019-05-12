using Distributions, Random
export smc2

function getProposal(theta; x0 = [0, 1.0, 0, 0])
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(theta)
hmm = HMM(armondhmmSimple, transll)
proposal = auxiliaryprop(armondhmmSimple, x0, approxtrans, approxll)
return proposal
end

function smc2(;observations::Matrix{Float}, Nthet::Int,
              N::Int, thin::Int, priors,
              parameterProp,
              resampling::Function=multinomialresampling,
              essthresh::Float=0.5
              )
#parameterProp should be a distributions object
    T   = size(observations, 2)
  Nx = N*thin #max number of particles for hidden state
d = 2
dim = 4 # dimx
  tunemat = zeros(d,d)
  xprior = zeros(Nx,dim)
  thetasamples = zeros(Nthet,d)
  thetasamplesTMP = zeros(Nthet,d)
  xparts = zeros(Nx,Nthet,dim)
  xpartsTMP = zeros(Nx,Nthet,dim)
  mll = zeros(Nthet)
  wtsthetEXP = ones(Nthet)
  wtsthet = zeros(Nthet)


  #sample prior and initialise marginal likelihood vector, parameter 
if isnothing(priors)
  #set prior
  v_plus_prior = TruncatedNormal(0.03, sqrt(10),0,Inf)
  v_minus_prior = TruncatedNormal(-0.03,sqrt(10),-Inf,0)
  alpha_prior = TruncatedNormal(0.01,sqrt(10000),0,Inf)
  kappa_prior = TruncatedNormal(0.05,sqrt(10000),0,Inf)
  tau_prior = Gamma(1/0.5,1/0.001) #shape vs rate paramterisation
  L_prior = TruncatedNormal(0.775,sqrt(0.121),0,Inf)
  p_coh_prior = Beta(2.5,1)
  p_icoh_prior = Beta(2,1)
else #different way to store to make more general?
  v_plus_prior = priors[1]
  v_minus_prior = priors[2]
  alpha_prior = priors[3]
  kappa_prior = priors[4]
  tau_prior = priors[5]
  L_prior = priors[6]
  p_coh_prior = priors[7]
  p_icoh_prior = priors[8]
end

#=
#TODO one or the other??
  for i=1:N
    for k=1:Nthet
        xparts[i,k,:] = x0
    end
  end
=#
  wts = ones(N)  #set state weights
  #apply priors for each parameter
  thetasamples[:,1] = log.(rand(p_coh_prior,Nthet))
  thetasamples[:,2] = log.(rand(p_coh_prior,Nthet))

#t=1 case!?
#    psf = ParticleSetTheta(Nthet,Nx,d, hmm.dimx, T)
#    ess = zeros(Nthet, T)
for k=1:Nthet
  ###NEED THE PROPOSAL TO BE DEPENDENT ON THETA
  th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, thetasamples[k,1], thetasamples[k,2], 0.775, 3.4)
  proposal = getProposal(th)
  #initialise state particles
    (p1,e1) = resample( Particles(
                            [proposal.mu0 + proposal.noise() for i in 1:N],
                            ones(N)/N),
                        essthresh )
    # store
    xparts[:,k,:] = p1  #TODO: check here
    ess[k,1]   = e1
end

for t=2:T
  for k=1:Nthet
    th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, thetasamples[k,1], thetasamples[k,2], 0.775, 3.4)
    logak = zeros(N)
    pkm1 = xparts[:,k,:]
    xk = similar(pkm1)
    for i in 1:N
        xk[i,:]    = proposal.mean(t, pkm1[i,:], observations[:,t-1], observations[:,t]) + proposal.noise()
        logak[i] = hmm.transloglik(t, pkm1[i,:], xk[i,:]) +
                    hmm.obsloglik(t, observations[:,t-1], observations[:,t], xk[i,:]) -
                    proposal.loglik(t, pkm1[i,:], observations[:,t-1], observations[:,t], xk[i,:])
    end
    Wk  = log.(wts) + logak
    Wk .-= minimum(Wk) # try to avoid underflows
    wk  = exp.(Wk)
#    wts = deepcopy(wk) #update unnormalised weights too?
    wk /= sum(wk)
    (pk, ek) = resample(Particles(xk,wk), essthresh, resampling)

    xparts[:,k,:] = pk #note not storing entire history of particles
    ess[k,t]   = ek
    println(pk, ek)

    wtsthetInc = log(sum(wts))-log(N)
    mll[k] += wtsthetInc
    wtsthet[k] += wtsthetInc
    essNum += exp(wtsthet[k]);
    essDenom += exp(wtsthet[k])*exp(wtsthet[k])
  end
  essC = essNum*essNum/essDenom #calculate ESS

if essC < essthresh*Nthet
println("ESS in theta is low so resample and move")
(pt, et) = resample(Particles(thetasamples,exp.(wtsthet)), essthresh, resampling)    
println(pt,et)

#TODO: put adaptive proposal in here
count = 0
#propose c* ~ q(.|c^k)
thetasamplesTMP = deepcopy(thetasamples)
xpartsTMP = deepcopy(xparts)
thetasamples = rand(parameterProp,Nthet)

  for k=1:Nthet  #each parameter particle through MH kernel
    wts = ones(N)  #set state weights
    th = thetaSimple(450, 0.008, 0.025, -0.015, 0.035, thetasamples[k,1], thetasamples[k,2], 0.775, 3.4)
    proposal = getProposal(th)
    #initialise state particles
    (p1,e1) = resample( Particles(
                            [proposal.mu0 + proposal.noise() for i in 1:N],
                            ones(N)/N),
                        essthresh )
    # store
    xparts[:,k,:] = p1  #TODO: check here

    logak = zeros(N)
    pkm1 = xparts[:,k,:]
    xk = similar(pkm1)
    for i in 1:N
        xk[i,:]    = proposal.mean(t, pkm1[i,:], observations[:,t-1], observations[:,t]) + proposal.noise()
        logak[i] = hmm.transloglik(t, pkm1[i,:], xk[i,:]) +
                    hmm.obsloglik(t, observations[:,t-1], observations[:,t], xk[i,:]) -
                    proposal.loglik(t, pkm1[i,:], observations[:,t-1], observations[:,t], xk[i,:])
    end
    Wk  = log.(wts) + logak
    Wk .-= minimum(Wk) # try to avoid underflows
    wk  = exp.(Wk)

	  qdiff = pdf(parameterProp,thetasamples[k,:]) - pdf(parameterProp,thetasamplesTMP[k,:]) #proposal ratio
	  priorCont = pdf(p_coh_prior,thetasamples[k,:]) - pdf(p_coh_prior,thetasampleTMP[k,:]) +
                      pdf(p_icoh_prior,thetasamples[k,:]) - pdf(p_icoh_prior,thetasamplesTMP[k,:]) #prior contribution
	  mllprop = deepcopy(Wk)
	  mllcurr = mll[k]
	  aprob = mllprop+priorCont-mllcurr+qdiff #evaluate acceptance ratio

	  u = rand()
	  if (log(u) < aprob) 
	    #Accept
	    mll[k] = mllprop  #update mll
          else 
            xparts = deepcopy(xpartsTMP)
            thetasamples = deepcopy(thetasamplesTMP)
	    count+=1  #track chain move
          end
	println(t,essC,count)
  end
 
end
end
end
