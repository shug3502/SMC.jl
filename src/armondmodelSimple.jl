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

function stochasticTransition(prob::Array{Float}, nStates::Int, u::Union{Array{Float},Float,Nothing})
#given probabilities of a transition to each of nStates, work out which one to switch to
    @assert length(prob) == nStates
    u = isnothing(u) ? rand() : u
    cumProb = cumsum(prob)
    @assert isapprox(cumProb[nStates],1.0)
    xk = zeros(nStates)
    for j = 1:nStates
        if (u<cumProb[j])
            xk[j] = 1
            break
        end
    end
    return xk
end
    
function armondModelSimple(th::Union{Nothing,thetaSimple}=nothing)
    th = (isnothing(th)) ? thetaSimple(450, 0.008, 0.025, -0.035, 0.015, 0.6, 0.9, 0.775, 2.0) : th #set default
    nSisters = 2
    nStates = 4
    angleTheta = 0 #for rotation when in 3D
    R = th.dt/th.tau*eye(nSisters) #variance matrix

    #p_coh and p_icoh instead as per Armond et al 2015 rather than reparameterized
    p_icoh = th.p_icoh
    p_coh = th.p_coh
    q_icoh = 1-p_icoh
    q_coh = 1-p_coh
    P = [p_icoh*p_icoh p_icoh*q_icoh p_icoh*q_icoh q_icoh*q_icoh;
        p_coh*q_coh p_coh*p_coh q_coh*q_coh p_coh*q_coh;
        p_coh*q_coh q_coh*q_coh p_coh*p_coh p_coh*q_coh;
        q_icoh*q_icoh p_icoh*q_icoh p_icoh*q_icoh p_icoh*p_icoh]

    function transmean(k::Int, xkm1::Union{Array{Int},Array{Float}}, u::Union{Array{Float},Nothing}, P::Array{Float64})
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev)
        prob = P[whichstateprev,:]
        xk = stochasticTransition(prob,nStates,u)
        @assert sum(xk) > 0 
        return xk
    end

    function odeUpdateMatrix(theta::thetaSimple)
        M = [(-theta.kappa - theta.alpha) theta.kappa -theta.v_plus -theta.v_plus -theta.v_minus -theta.v_minus; 
            theta.kappa (-theta.kappa - theta.alpha) theta.v_plus theta.v_minus theta.v_plus theta.v_minus]
        return M
    end

    function odeUpdateVector(theta::thetaSimple)
        mu = [theta.kappa*theta.L*cos(angleTheta); -theta.kappa*theta.L*cos(angleTheta)]
        return mu
    end

    function obsmean(k::Int, xk::Union{Array{Int},Array{Float}}, ykm1::Array{Float}, theta::thetaSimple)    
    state = [ykm1; xk]
    #forward euler step
    state[1:nSisters] += theta.dt*odeUpdateMatrix(theta)*state + theta.dt*odeUpdateVector(theta)
    return state[1:nSisters]
    end

    function transloglik(k::Int,xkm1::Union{Array{Int},Array{Float}},xk::Union{Array{Int},Array{Float}},P::Array{Float})
    whichstateprev = findfirst(w -> w>0, xkm1)
    whichstatenext = findfirst(w -> w>0 , xk)
    @assert !isnothing(whichstatenext) "oops: the next state is $whichstatenext since we had $xkm1 and then $xk" 
    transition_prob = P[whichstateprev,whichstatenext] #probability of getting between hidden states for xkm1 and xk   
    return log(transition_prob)
    end

    function approxtransmean(k::Int, xkm1::Union{Array{Int},Array{Float}}, ykm1::Array{Float}, yk::Array{Float}, theta::thetaSimple,
                             u::Union{Nothing,Array{Float},Float}, P::Array{Float})
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev) "oops the previous state was $xkm1"
        transition_prob = P[whichstateprev,:]
        Q = Array{Float}(undef, length(transition_prob))
        copyto!(Q, log.(transition_prob)) #this needs to be reweighted appropriately due to conditioning
        for j=1:nStates
            xk = zeros(nStates)
            xk[j] = 1 #potential new state
            Q[j] += logpdf(MvNormal(ykm1 + theta.dt*odeUpdateMatrix(theta)*[ykm1; xk] + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),yk) #TODO could be more general for non isotropic gaussian
        end
        b = maximum(Q)
        prob = exp.(Q .- b) #subtract for numerical stability
        prob ./= sum(prob) #normalise
        xk = stochasticTransition(prob,nStates,u)
        return xk
    end

    function approxloglik(k::Int, xkm1::Union{Array{Int},Array{Float}}, ykm1::Array{Float}, yk::Array{Float},
                          xk::Union{Array{Int},Array{Float}}, theta::thetaSimple, P::Array{Float})
        whichstateprev = findfirst(w -> w>0, xkm1)
        whichstatenext = findfirst(w -> w>0 , xk)
        @assert !isnothing(whichstateprev)
        @assert !isnothing(whichstatenext)
        transition_prob = P[whichstateprev,:]
        Q = Array{Float}(undef, length(transition_prob))
        copyto!(Q, log.(transition_prob)) #this needs to be reweighted appropriately due to conditioning
        for j=1:nStates
            xk = zeros(nStates)
            xk[j] = 1 #potential new state
            Q[j] += logpdf(MvNormal(ykm1 + theta.dt*odeUpdateMatrix(theta)*[ykm1; xk] + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),yk) #TODO could be more general for non isotropic gaussian
        end
        b = maximum(Q)
        prob = exp.(Q .- b) #subtract for numerical stability
        prob ./= sum(prob) #normalise
        return log(prob[whichstatenext])
    end

    armondhmmSimple = DiscreteState((k,xk,u=nothing) -> transmean(k,xk,u,P), (k,xk,ykm1) -> obsmean(k,xk,ykm1,th), R, 4, 2)
    transll(k,xkm1,xk) = transloglik(k,xkm1,xk,P)
    approxll(k, xkm1, ykm1, yk, xk) = approxloglik(k, xkm1, ykm1, yk, xk, th, P)
    approxtrans(k, xkm1, ykm1, yk, u=nothing) = approxtransmean(k, xkm1, ykm1, yk, th, u, P)

    return armondhmmSimple, transll, approxtrans, approxll 
end


