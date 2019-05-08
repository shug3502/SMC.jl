using Distributions

export
    thetaSimple,
    armondModelSimple,
    stochasticTransition

#set model parameters
struct thetaSimple
    tau::Float64
    alpha::Float64
    kappa::Float64
    v_minus::Float64
    v_plus::Float64
    p_icoh::Float64
    p_coh::Float64
    L::Float64
    dt::Float64
end

function stochasticTransition(prob::Array, nStates::Int)
#given probabilities of a transition to each of nStates, work out which one to switch to
    @assert length(prob) == nStates
    cumProb = cumsum(prob)
    @assert isapprox(cumProb[nStates],1.0)
    u = rand()
    xk = zeros(nStates)
    for j = 1:nStates
        if (u<cumProb[j])
            xk[j] = 1
            break
        end
    end
    return xk
end
    
function armondModelSimple(th=nothing)
    th = (isnothing(th)) ? thetaSimple(450, 0.008, 0.025, -0.035, 0.015, 0.6, 0.9, 0.775, 3.4) : th #set default
    nSisters = 2
    nStates = 4
    angleTheta = 0 #for rotation when in 3D
    R = th.dt/th.tau*eye(nSisters) #variance matrix

    #p_coh and p_icoh instead as per Armond et al 2015 rather than reparameterized
    p_icoh = th.p_icoh
    p_coh = th.p_icoh
    q_icoh = 1-p_icoh
    q_coh = 1-p_coh
    P = [p_icoh*p_icoh p_icoh*q_icoh p_icoh*q_icoh q_icoh*q_icoh;
        p_coh*q_coh p_coh*p_coh q_coh*q_coh p_coh*q_coh;
        p_coh*q_coh q_coh*q_coh p_coh*p_coh p_coh*q_coh;
        q_icoh*q_icoh p_icoh*q_icoh p_icoh*q_icoh p_icoh*p_icoh]

    function transmean(k, xkm1::Array, theta)
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev)
        prob = P[whichstateprev,:]
        xk = stochasticTransition(prob,nStates)
        @assert sum(xk) > 0 
        return xk
    end

    function odeUpdateMatrix(theta)
        M = [-theta.v_plus -theta.v_plus -theta.v_minus -theta.v_minus (-theta.kappa - theta.alpha) -theta.kappa; 
            theta.v_plus theta.v_minus theta.v_plus theta.v_minus theta.kappa (-theta.kappa - theta.alpha)]
        return M
    end

    function odeUpdateVector(theta)
        mu = [theta.kappa*theta.L*cos(angleTheta); -theta.kappa*theta.L*cos(angleTheta)]
        return mu
    end

    function obsmean(k, xk::Array, ykm1::Array, theta)    
    state = [ykm1; xk]
    #forward euler step
    state[1:nSisters] += theta.dt*odeUpdateMatrix(theta)*state + theta.dt*odeUpdateVector(theta)
    return state[1:nSisters]
    end

    function transloglik(k,xkm1,xk,theta)
    #TODO: check normalizing constants in cases where we care about infering process/observation noise
    whichstateprev = findfirst(w -> w>0, xkm1)
    whichstatenext = findfirst(w -> w>0 , xk)
    @assert !isnothing(whichstatenext)
    transition_prob = P[whichstateprev,whichstatenext] #probability of getting between hidden states for xkm1 and xk   
    return log(transition_prob)
    end

    function approxtransmean(k, xkm1, ykm1, yk, theta)
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev)
        prob = P[whichstateprev,:]
        Q = deepcopy(log(prob)) #this needs to be reweighted appropriately due to conditioning
        for j=1:nStates
            xk = zeros(nStates)
            xk[j] = 1 #potential new state
            Q[j] += logpdf(MvNormal(ykm1 + theta.dt*odeUpdateMatrix(theta)*[ykm1; xk] + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),yk) #TODO could be more general for non isotropic gaussian
        end
        b = max(Q)
        prob = exp(Q - b) #subtract for numerical stability
        prob ./= sum(prob) #normalise
        xk = stochasticTransition(prob,nStates)
        return xk
    end

    function approxloglik(k, xkm1, ykm1, yk, xk, theta)
        whichstateprev = findfirst(w -> w>0, xkm1)
        whichstatenext = findfirst(w -> w>0 , xk)
        @assert !isnothing(whichstateprev)
        @assert !isnothing(whichstatenext)
        prob = P[whichstateprev,:]
        Q = deepcopy(log(prob)) #this needs to be reweighted appropriately due to conditioning
        for j=1:nStates
            xk = zeros(nStates)
            xk[j] = 1 #potential new state
            Q[j] += logpdf(MvNormal(ykm1 + theta.dt*odeUpdateMatrix(theta)*[ykm1; xk] + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),yk) #TODO could be more general for non isotropic gaussian
        end
        b = max(Q)
        prob = exp(Q - b) #subtract for numerical stability
        prob ./= sum(prob) #normalise
        return log(prob[whichstatenext])
    end

    armondhmmSimple = DiscreteState((k,xk) -> transmean(k,xk,th), (k,xk,ykm1) -> obsmean(k,xk,ykm1,th), R, 4, 2)
    transll(k,xkm1,xk) = transloglik(k,xkm1,xk,th)
    approxll(k, xkm1, ykm1, yk, xk) = approxloglik(k, xkm1, ykm1, yk, xk, th)
    approxtrans(k, xkm1, ykm1, yk) = approxtransmean(k, xkm1, ykm1, yk, th)

    return armondhmmSimple, transll, approxtrans, approxll 
end


