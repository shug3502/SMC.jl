export particlegibbs,
       gibbssampler

function particlegibbs(hmm::HMM, observations::Matrix{Float},
                       initialisationFn::Array, priors::Array,
                       N::Int, nIter::Int, dimParams::Int;
                       model::String="Simple", dt::Float=2.0,
                       x0::Array=[0, 1, 0, 0],
                       resampling::Function=multinomialresampling,
                       essthresh::Float=0.5, u=nothing,
                       resampler::Function=resample,
                       filterMethod::String="Aux",
                       printFreq::Int=1000,
                       maxRejects::Int=1000,
                       burnin::Int=round(Int,nIter/10)
)

#initialize theta, initialize x_{1:T}
############################
K = size(observations,2)
hiddenstates = zeros(hmm.dimx, K, nIter)
hiddenstates[2,:,1] = ones(K)
c = zeros(dimParams,nIter)
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
  if n%printFreq == 0
      println("Iter: ", n)
    end

  #run csmc to draw x_{1:T} given theta
##########################
  #update proposal
  th, prop, hmm = constructProposal(c[:,1], model, dt, "Aux",x0)
  retained_particle = [hiddenstates[:,k,(n-1)] for k in 1:K]
  new_state = csmc(retained_particle, hmm, observations, N, prop,
       resampling=resampling, essthresh=essthresh, u=u, resampler=resampler)
  for k=1:K
    hiddenstates[:,k,n] = new_state[k]
  end
##########################
  #draw theta given x_{1:T} based on previous methods 
  cTemp = copy(c[:,n-1])
  cTemp[1] = gibbsUpdateTau(cTemp,hiddenstates[:,:,n],observations)
  cTemp[4:5] = gibbsUpdateVpm(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[3] = gibbsUpdateKappa(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[8] = gibbsUpdateL(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[2] = gibbsUpdateAlpha(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[6:7] = gibbsUpdatePcohicoh(cTemp,hiddenstates[:,:,n],observations)
  c[:,n] = cTemp
end
    states = mapslices(x -> findfirst(w -> w>0, x),hiddenstates,dims=1)
    actRate = 1
return (transpose(c[:,(burnin+1):nIter]), transpose(dropdims(states,dims=1)[:,(burnin+1):nIter]), actRate)
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
    psf.p[1] = p1;
  #compute initial weights here: previously have just used uniform weights here, seems this has been fine
  for k = 2:K
    pkm1 = psf.p[k-1]
    obsk = observations[:,k]
    obskm1 = observations[:,k-1] #only used properly for aux filter
    logak = zeros(N)
    xk    = similar(pkm1.x)
    prob_J = zeros(N)
    #sample x_{1:t}^i with replacement from the previous set of weighted particles
    (pk,~,ek) = resampler(pkm1, essthresh, resampling, 0, u[K*(N+1)-(k-2)])
    for i = 1:(N-1)
      #draw J with appropriate weights - may need different form if not markov 
      prob_J[i] = pkm1.w[i]*hmm.transloglik(k, pkm1.x[i], retained_particle[k])
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
new_draw = similar(retained_particle)
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

    function odeUpdateMatrix(c::Array)
        M = [(-c[3] - c[2]) c[3] -c[5] -c[5] -c[4] -c[4]; 
            c[3] (-c[3] - c[2]) c[5] c[4] c[5] c[4] ]
        return M
    end

    function odeUpdateVector(c::Array; angleTheta::Float=0.0)
        mu = [c[3]*c[8]*cos(angleTheta);
              -c[3]*c[8]*cos(angleTheta)]
        return mu
    end

function gibbsUpdateVpm(c::Array, x::Array, observations::Array;
                        prior_musd::Array = [0.03 sqrt(10); -0.03 sqrt(10)],
                        dt::Float = 2.0, angleTheta::Float=0.0,
                        maxRejects::Int=1000)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prio$
  ###############
  K = size(x,2)

  #TODO: may need to exclude either first or last hidden state to get K-1 pts
  num_pm_pts = [sum(x[1:3,:].*[2*ones(1,K); ones(2,K)]), #plus: count ++ twice
              sum(x[2:4,:].*[ones(2,K); 2*ones(1,K)])] #minus

  tauv = zeros(2);
  v_pm = [-1.0,1.0]; #this should not satisfy the next condition so should enter loop
  RR = [1 1 0 0;
        1 0 1 0;
        0 0 1 1;
        0 1 0 1]
    
for ipm=1:2 #plus or minus
  tauv[ipm]=1/prior_musd[ipm,2]^2 + c[1]*(num_pm_pts[ipm])
  aux_sum = 0
        
  for k=2:K
    RRx = RR*x[:,k]
    for jSister = 1:2
      if sum(RRx[range(1+2*(ipm-1), stop=2*ipm)]) > 0 #determine whether this data pt contributes or not
      aux_sum += (-1)^jSister*(observations[jSister,k] - observations[jSister,k-1] +
               c[2]*observations[jSister,k-1]) - c[3]*(observations[1,k-1] -
               observations[2,k-1] - c[8]*cos(angleTheta))
      end
    end
  end
  count = 0
  while (-1)^ipm*v_pm[ipm] > 0 #enforce constraints
    count += 1
    #keep redrawing
            #TODO: check scaling with dt in NJBs code
    v_pm[ipm] = (-1)^(ipm)*(prior_musd[ipm,1]/prior_musd[ipm,2]^2 +
            c[1]*aux_sum)/tauv[ipm] + randn()/sqrt(tauv[ipm])
    if count > maxRejects
      @info "v+-: $v_pm" maxlog=20                 
      @warn "Seem to be stuck somewhere and rejecting lots of v+- proposals: rejecting proposal" maxlog=20
      v_pm = [c[5], c[4]]
    end
  end
end
  return v_pm[2:-1:1] #reverse since need [v_minus, v_plus] not [v_plus, v_minus]
end

function gibbsUpdateKappa(c::Array, x::Array, observations::Array;
                        prior_musd::Array = [0.05 sqrt(10000)],
                        dt::Float = 2.0, angleTheta::Float=0.0,
                        maxRejects::Int=1000)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prio$
  ###############
  K = size(x,2)

  taukappa=1/prior_musd[2]^2 + 2*c[1]*sum(
  [(observations[1,k] - observations[2,k] - c[8]*cos(angleTheta))^2 for k=1:(K-1)])

  RS = [c[4] c[4] c[5] c[5]; #+ + - -
      c[4] c[5] c[4] c[5]]; #+ - + -
  aux_sum = 0        
  for k=2:K
    for jSister = 1:2
    v = RS[jSister,:]' * x[:,k] #decide whether vplus or vminus
    aux_sum += ((-1)^jSister*(observations[jSister,k] - observations[jSister,k-1] +
               c[2]*observations[jSister,k-1]) - v )*(observations[1,k-1] -
               observations[2,k-1] - c[8]*cos(angleTheta))
    end
  end
  kappa = -1
  count = 0
  while kappa < 0 #ensure positivity in proposed value
    count +=1
    kappa = (prior_musd[1]/prior_musd[2]^2 + c[1]*aux_sum)/taukappa + randn()/sqrt(taukappa)
    if count > maxRejects
      @info "kappa: $kappa" maxlog=20
      @warn "Stuck somewhere with bad proposals for kappa: rejecting proposal" maxlog=20
      kappa = c[3]
    end
  end
  return kappa
end

function gibbsUpdateL(c::Array, x::Array, observations::Array;
                        prior_musd::Array = [0.775 sqrt(0.0121)],
                        dt::Float = 2.0, angleTheta::Float=0.0,
                        maxRejects::Int=1000)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prio$
  ###############
  K = size(x,2)
  tauL=1/prior_musd[2]^2 + 2*c[1]*c[3]^2*sum([cos(angleTheta)^2 for k=1:(K-1)]) #K-1 or K pts in sum

  RS = [c[4] c[4] c[5] c[5]; #+ + - -
      c[4] c[5] c[4] c[5]]; #+ - + -
  aux_sum = 0        
  for k=2:K
    for jSister = 1:2
      v = RS[jSister,:]' * x[:,k] #decide whether vplus or vminus
      aux_sum += cos(angleTheta)*((-1)^(jSister+1)*(observations[jSister,k] - observations[jSister,k-1] +
               c[2]*observations[jSister,k-1]) + v + c[3]*(observations[1,k-1] - observations[2,k-1]))
    end
  end
  L = -1
  count = 0
  while L < 0 #ensure positivity in proposed value
    count +=1
    L = (prior_musd[1]/prior_musd[2]^2 + c[1]*c[3]*aux_sum)/tauL + randn()/sqrt(tauL)
    if count > maxRejects
      @info "L: $L" maxlog=20
      @warn "Stuck somewhere with bad proposals for L: rejecting proposal" maxlog=20
      L = c[8]
    end
  end
  return L
end

function gibbsUpdateAlpha(c::Array, x::Array, observations::Array;
                        prior_musd::Array = [0.01 sqrt(10000)],
                        dt::Float = 2.0, angleTheta::Float=0.0,
                        maxRejects::Int=1000)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prio$
  ###############
  K = size(x,2)
  tauAlpha=1/prior_musd[2]^2 + c[1]*sum(observations[:,1:(K-1)].^2) #K-1 or K pts in sum

  RS = [c[4] c[4] c[5] c[5]; #+ + - -
      c[4] c[5] c[4] c[5]]; #+ - + -
  aux_sum = 0        
  for k=2:K
    for jSister = 1:2
      v = RS[jSister,:]' * x[:,k] #decide whether vplus or vminus
      aux_sum += observations[jSister,k-1]*(observations[jSister,k] - observations[jSister,k-1] +
               + (-1)^(jSister+1)*v + (-1)^(jSister+1)*c[3]*(
                 observations[1,k-1] - observations[2,k-1] - c[8]*cos(angleTheta)))
    end
  end
  alpha = -1
  count = 0
  while alpha < 0 #ensure positivity in proposed value
    count +=1
    alpha = (prior_musd[1]/prior_musd[2]^2 - c[1]*aux_sum)/tauAlpha + randn()/sqrt(tauAlpha)
    if count > maxRejects
      @info "alpha: $alpha" maxlog=20
      @warn "Stuck somewhere with bad proposals for alpha: rejecting proposal" maxlog=20
      alpha = c[2]
      break
    end
  end
  return alpha
end

function gibbsUpdatePcohicoh(c::Array, x::Array, observations::Array;
                        prior_ab::Array = [2.5 1; 2 1],
                        dt::Float = 2.0, angleTheta::Float=0.0)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prio$
  ###############
  K = size(x,2)
  is_coherent = [0, 1.0, 1.0, 0]'*x
  is_incoherent = [1.0, 0, 0, 1.0]'*x
  num_coherent = sum(is_coherent)
  num_incoherent = sum(is_incoherent)

  diff_coh = [is_coherent[k] - is_coherent[k-1] for k=2:K] #use this to detect changes
  K_coh = sum(diff_coh .< 0) #switch to incoherence
  K_icoh = sum(diff_coh .> 0) #switch to coherence

  p_coh = rand(Beta(prior_ab[1,1] + num_coherent - K_coh,
                    prior_ab[1,2] + K_coh))
  p_icoh = rand(Beta(prior_ab[2,1] + num_incoherent - K_icoh,
                    prior_ab[2,2] + K_icoh)) 
  return [p_coh, p_icoh]
end

function gibbsUpdateQana(c::Array, x::Array, observations::Array;
                        prior_ab::Array = [1 100],
                        dt::Float = 2.0, angleTheta::Float=0.0)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior. (prio$
  ###############
  K = size(x,2)
  is_ana = [0, 0, 0, 0, 1.0]'*x
  num_ana = sum(is_ana .< 1.0) #number of time pts up to 1st switch
  K_ana = is_ana[k] #assuming anaphase state is absorbing, looking at final state indicates whether switched to anaphase during observed period or not

  q_ana = rand(Beta(prior_ab[1] + K_ana,
                    prior_ab[2] + num_ana - K_ana))
  return q_ana
end

function gibbsUpdateVana(c::Array, x::Array, observations::Array;
                        prior_ab::Array = [1 100],
                        dt::Float = 2.0, angleTheta::Float=0.0)
  #based on supplementary material from Armond et al 2015
  #check how gamma distn is paramterized in Julia
  #prior_cd gives the parameterization of the conjugate prior.
  ###############
  K = size(x,2)

num_pm_pts = [sum(x[1:3,:].*[2*ones(1,K); ones(2,K)]), #plus: count ++ twice
              sum(x[2:4,:].*[ones(2,K); 2*ones(1,K)])] #minus

  tauv = zeros(2);
  v_pm = [-1.0,1.0]; #this should not satisfy the next condition so should enter loop
  RR = [1 1 0 0;
        1 0 1 0;
        0 0 1 1;
        0 1 0 1]

for ipm=1:2 #plus or minus
  tauv[ipm]=1/prior_musd[ipm,2]^2 + c[1]*(num_pm_pts[ipm])
  aux_sum = 0

  for k=2:K
    RRx = RR*x[:,k]
    for jSister = 1:2
      if sum(RRx[range(1+2*(ipm-1), stop=2*ipm)]) > 0 #determine whether this data pt contributes or$
      aux_sum += (-1)^jSister*(observations[jSister,k] - observations[jSister,k-1] +
               c[2]*observations[jSister,k-1]) - c[3]*(observations[1,k-1] -
               observations[2,k-1] - c[8]*cos(angleTheta))
      end
    end
  end
  count = 0
  while (-1)^ipm*v_pm[ipm] > 0 #enforce constraints
    count += 1
    #keep redrawing
            #TODO: check scaling with dt in NJBs code
    v_pm[ipm] = (-1)^(ipm)*(prior_musd[ipm,1]/prior_musd[ipm,2]^2 +
            c[1]*aux_sum)/tauv[ipm] + randn()/sqrt(tauv[ipm])
    if count > maxRejects
      @info "v+-: $v_pm" maxlog=20
      @warn "Seem to be stuck somewhere and rejecting lots of v+- proposals: rejecting proposal" max$
      v_pm = [c[5], c[4]]
    end
  end
end

  
  return v_ana
end

function gibbsUpdateHiddenStates(c::Array, x::Array, observations::Array, block::Array;
                        dt::Float = 2.0)
  #update for a block of time points 

    p_icoh = c[6]
    p_coh = c[7]
    q_icoh = 1-c[6]
    q_coh = 1-c[7]
    P = [p_icoh*p_icoh p_icoh*q_icoh p_icoh*q_icoh q_icoh*q_icoh;
        p_coh*q_coh p_coh*p_coh q_coh*q_coh p_coh*q_coh;
        p_coh*q_coh q_coh*q_coh p_coh*p_coh p_coh*q_coh;
        q_icoh*q_icoh p_icoh*q_icoh p_icoh*q_icoh p_icoh*p_icoh]
  nStates = size(x,1)
  xindx = mapslices(y -> findfirst(y .> 0),x,dims=1)  #convert from binary to index   
  Q = zeros(nStates)
  for b = block
      for i=1:nStates
        state = [observations[:,b]; zeros(nStates)]
        state[2+i] = 1.0
        model_dx = odeUpdateMatrix(c)*state*dt .+ odeUpdateVector(c) .* dt
        if (b>1) & (b < size(observations,2))
          Q[i] = log(P[xindx[b-1],i]) + logpdf(MvNormalCanon(model_dx, c[1]),observations[:,b+1] .- observations[:,b]) + log(P[i,xindx[b+1]])
        elseif b==1
          #treat differently
          Q[i] = logpdf(MvNormalCanon(model_dx,c[1]),observations[:,b+1] .- observations[:,b]) + log(P[i,xindx[b+1]])
        elseif b==size(observations,2)
          #treat differently
          Q[i] = log(P[xindx[b-1],i]) + logpdf(MvNormalCanon(model_dx,c[1]),observations[:,b] .- observations[:,b-1])
        end
      end
    Q .-= minimum(Q)
    prob = exp.(Q)
    prob ./= sum(prob)
    @assert isapprox(sum(prob), 1.0)
    xindx[b] = rand(Categorical(prob))
    x[xindx[b],b] = 1.0
    x[setdiff(1:nStates,xindx[b]),b] .= 0.0
  end
return x
end

function gibbssampler(hmm::HMM, observations::Matrix{Float},
                       initialisationFn::Array, priors::Array,
                       nIter::Int, dimParams::Int;
                       model::String="Simple", dt::Float=2.0,
                       x0::Array=[0, 1, 0, 0],
                       u=nothing,
                       printFreq::Int=1000,
                       maxRejects::Int=1000,
                       burnin::Int=round(Int,nIter/10)
)

#initialize theta, initialize x_{1:T}
############################
K = size(observations,2)
hiddenstates = zeros(hmm.dimx, K, nIter)
hiddenstates[2,:,1] = ones(K)
c = zeros(dimParams,nIter)
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
  if n%printFreq == 0
      println("Iter: ", n)
    end

  #run csmc to draw x_{1:T} given theta
##########################
  #proposal for hidden states
  @info hiddenstates[:,:,n-1] maxlog=10
  statesTemp = copy(hiddenstates[:,:,n-1])
  for b = 1:K
    statesTemp = gibbsUpdateHiddenStates(c[:,n-1], statesTemp,
                      observations, [b], dt=dt)
  end
  @info statesTemp maxlog=10
  hiddenstates[:,:,n] = statesTemp 
##########################
  #draw theta given x_{1:T} based on previous methods 
  cTemp = copy(c[:,n-1])
  cTemp[1] = gibbsUpdateTau(cTemp,hiddenstates[:,:,n],observations)
  cTemp[4:5] = gibbsUpdateVpm(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[3] = gibbsUpdateKappa(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[8] = gibbsUpdateL(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[2] = gibbsUpdateAlpha(cTemp,hiddenstates[:,:,n],observations,maxRejects=maxRejects)
  cTemp[6:7] = gibbsUpdatePcohicoh(cTemp,hiddenstates[:,:,n],observations)
  c[:,n] = cTemp
end
    states = mapslices(x -> findfirst(w -> w>0, x),hiddenstates,dims=1)
    actRate = 1
return (transpose(c[:,(burnin+1):nIter]), transpose(dropdims(states,dims=1)[:,(burnin+1):nIter]), actRate)
end

