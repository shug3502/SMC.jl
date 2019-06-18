export particlegibbs

function particlegibbs(hmm::HMM, observations::Matrix{Float}, N::Int, nIter::Int, dimParams::Int;
                       model::String="Simple", dt::Float=2.0,
                       resampling::Function=multinomialresampling,
                       essthresh::Float=0.5, u=nothing,
                       resampler::Function=resample
)

#initialize theta, initialize x_{1:T}
############################
K = size(observations,2)
hiddenstates = zeros(hmm.dimx, K, numIter)
hiddenstates[2,:,1] = ones(K)
c = zeros(dimParams,numIter)
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
##########################

for n=2:nIter
  #run csmc to draw x_{1:T} given theta
##########################
  #update proposal
  th, prop, hmm = constructProposal(c[:,1], model, dt)
  retained_particle = [hiddenstates[:,k,(n-1)] for k in 1:K]
  new_state = csmc(retained_particle, hmm, observations, N, proposal,
       resampling=resampling, essthresh=essthresh, u=u, resampler=resampler)
  for k=1:K
    hiddenstates[:,k,n] = new_state[k]
  end
##########################
  #draw theta given x_{1:T} based on previous methods 

end
return (transpose(c), transpose(hiddenstates))
end

function csmc(retained_particle, hmm::HMM, observations::Matrix{Float}, N::Int,
                        proposal::Proposal;
                        resampling::Function=multinomialresampling,
                        essthresh::Float=0.5, u=nothing,
                        resampler::Function=resample
)
  #see review paper Naesseth et al 2019 "Elements of Sequential Monte Carlo"

    K = size(observations,2)
    u = isnothing(u) ? rand(K*(N+1)) : u #draw however many random numbers are needed
    # particle set filter (storage)
    psf = ParticleSet(N, hmm.dimx, K)
    ess = zeros(K)
    ev = 0
    #draw x_1^i
    (p1,~,e1) = resampler( Particles(
                            [proposal.mu0 + proposal.noise() for i in 1:N],
                            ones(N)/N),
                        essthresh, resampling, 0, u[1])

  p1.x[N] = retained_particle[1] #set for Nth particle at first time pt
  #compute initial weights here: previously have just used uniform weights here, seems this has been fine

  for k = 2:K
    pkm1 = psf.p[k-1]
    obsk = observations[:,k]
    obskm1 = observations[:,k-1] #only used properly for aux filter
    logak = zeros(N)
    xk    = similar(pkm1.x)
    prob_J = zeros(N)
    for i = 1:(N-1)
      #sample x_{1:t}^i with replacement from the previous set of weighted particles
      (pk,~,ek) = resampler(pkm1, essthresh, resampling, 0, u[K*(N+1)-(k-2)])
      #draw J with appropriate weights - may need different form if not markov      
      prob_J = pkm1.w[i]*hmm.transloglik(k, pkm1.x[i], retained_particle[k])
    end
    prob_J /= sum(prob_J)
    J = rand(Categorical(prob_J)) #TODO: work out how best to use deterministic randomness here
    #set Nth particle based on this
    for kk = 1:(k-1)
      psf.p[kk].x[N] = psf.p[kk].x[J] #I think this is what it means - might want to check
    end
    for i=1:(N-1)
      #propogate particles
      #simulate x_t^i from auxiliary proposal for i=1:(N-1)
      pk.x[i]    = proposal.mean(k, pk.x[i], obskm1, obsk, u[(k-1)*N+i]) + proposal.noise(k,u[(k-1)*N+i]) #pass a single random number to the fwd simulation
      logak[i] = hmm.transloglik(k, pkm1.x[i], pk.x[i]) +
                        hmm.obsloglik(k, obskm1, obsk, pk.x[i]) -
                        proposal.loglik(k, pkm1.x[i], obskm1, obsk, pk.x[i])
    end
    #set x_t^N
    pk.x[N] = retained_particle[k]

    #weight all the particles for i=1:N    
    Wk  = log.(pkm1.w) + logak
    ev += sum(logak)/N #to compute the likelihood/evidence for p(y|c)
    Wk .-= minimum(Wk) # try to avoid underflows
    wk  = exp.(Wk)
    wk /= sum(wk)
    pk.w = copy(wk)
    
    #concatenate to current particles
    psf.p[k] = pk
    ##########      
  end
#draw one of the final particles to use as next reference path
draw_idx = rand(Categorical(psf.p[K].w))
new_draw = similar(pkm1.x)
for k=1:K
  new_draw[k] = psf.p[k].x[draw_idx]
end
return new_draw
end

function gibbsUpdateTau(c::Array, x::Array, observations::Array;
                        prior_cd::Array = [0.5,0.001],
                        dt::Float = 2.0)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prior should be gamma so that this gibbs update will work properly)
  ###############
  K = size(x,2)
  gt = Gamma(prior_cd[1]+(K-1),1)
  total_err = 0
  for k=2:K
    state = [observations[:,k-1]; x[:,k]]
    model_diff = dt*odeUpdateMatrix(c)*state + dt*odeUpdateVector(c)
    obs_diff = observations[:,k] .- observations[:,k-1]
    total_err += sum((obs_diff .- model_diff).^2)
  end
  tau = rand(gt) / (prior_cd[2]+0.5*total_err)
  return tau
end

    function odeUpdateMatrix(c::array)
        M = [(-c[3] - c[2]) c[3] -c[4] -c[5] -c[4] -c[4]; 
            c[3] (-c[3] - c[2]) c[5] c[4] c[5] c[4] ]
        return M
    end

    function odeUpdateVector(c::array; angleTheta::Float=0.0)
        mu = [c[3]*c[8]*cos(angleTheta);
              -c[3]*c[8]*cos(angleTheta)]
        return mu
    end

function gibbsUpdateVpm(c::Array, x::Array, observations::Array;
                        prior_musd::Array = [0.03, 10; -0.03, 10],
                        dt::Float = 2.0)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prio$
  ###############
  K = size(x,2)
  gt = Gamma(prior_cd[1]+(K-1),1)
  total_err = 0
  for k=2:K
    state = [observations[:,k-1]; x[:,k]]
    model_diff = dt*odeUpdateMatrix(c)*state + dt*odeUpdateVector(c)
    obs_diff = observations[:,k] .- observations[:,k-1]
    total_err += sum((obs_diff .- model_diff).^2)
  end
  tau = rand(gt) / (prior_cd[2]+0.5*total_err)

tauv=issqr+tau*(sum(negindices_s1)+sum(negindices_s2));
vmiprop=(-mu*issqr+tau*(-sum(dx(negindices_s1,1)+f(negindices_s1)+alpha*x1(negindices_s1))+sum(dx(negindices_s2,2)-f(negindices_s2)+alpha*x2(negindices_s2))))/tauv+randn(1)/sqrt(tauv);

  return tau
end

