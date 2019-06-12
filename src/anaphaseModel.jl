using Distributions

export
    thetaAnaphase,
    anaphaseModel

#For the anaphase model we will assume that we draw a time pt randomly in {1, ..., T} at which
# the anaphase transition occurs. After that we have different dynamics.    

#set model parameters
struct thetaAnaphase <: AbstractTheta
    tau::Float
    alpha::Float
    kappa::Float
    v_minus::Float
    v_plus::Float
    v_ana::Float
    p_icoh::Float
    p_coh::Float
    q_ana::Float #prob of transition to anaphase in a time step
    L::Float
    dt::Float
end

function anaphaseModel(th::Union{Nothing,AbstractTheta}=nothing)
    th = (isnothing(th)) ? thetaAnaphase(450, 0.008, 0.025, -0.035, 0.015, 0.020, 0.6, 0.9, 0.02, 0.775, 2.0) : th #set default
    nSisters = convert(Int,2)
    nStates = convert(Int,5)
    angleTheta = convert(Float,0) #for rotation when in 3D
    R = th.dt/th.tau*eye(nSisters) #variance matrix

    #p_coh and p_icoh instead as per Armond et al 2015 rather than reparameterized
    p_icoh = th.p_icoh
    p_coh = th.p_coh
    q_ana = th.q_ana
    q_icoh = 1-p_icoh
    q_coh = 1-p_coh
    p_ana = 1-q_ana
    P = [p_icoh*p_icoh*p_ana p_icoh*q_icoh*p_ana p_icoh*q_icoh*p_ana q_icoh*q_icoh*p_ana q_ana;
        p_coh*q_coh*p_ana p_coh*p_coh*p_ana q_coh*q_coh*p_ana p_coh*q_coh*p_ana q_ana;
        p_coh*q_coh*p_ana q_coh*q_coh*p_ana p_coh*p_coh*p_ana p_coh*q_coh*p_ana q_ana;
        q_icoh*q_icoh*p_ana p_icoh*q_icoh*p_ana p_icoh*q_icoh*p_ana p_icoh*p_icoh*p_ana q_ana;
        0 0 0 0 1.0]

    function transmean(k::Int, xkm1::Union{Array{Int},Array{Float}}, u::Union{Array{Float},Float,Nothing}, P::Array{Float})
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev)
        prob = P[whichstateprev,:]
        xk = stochasticTransition(prob,nStates,u)
        @assert sum(xk) > 0 
        return xk
    end

    function odeUpdateMatrix(theta::AbstractTheta)
        M = [(-theta.kappa - theta.alpha) theta.kappa -theta.v_plus -theta.v_plus -theta.v_minus -theta.v_minus 0; 
            theta.kappa (-theta.kappa - theta.alpha) theta.v_plus theta.v_minus theta.v_plus theta.v_minus 0]
        return M
    end

    function odeUpdateVector(theta::AbstractTheta)
        mu = [theta.kappa*theta.L*cos(angleTheta); -theta.kappa*theta.L*cos(angleTheta)]
        return mu
    end

    function obsmean(k::Int, xk::Union{Array{Int},Array{Float}}, ykm1::Array{Float}, theta::AbstractTheta)    
    state = [ykm1; xk]
    #forward euler step
    if xk[5]>0
        #anaphase transition has occurred
        state[1:nSisters] += theta.dt.*[theta.v_ana; -theta.v_ana]
    else
        state[1:nSisters] += theta.dt*odeUpdateMatrix(theta)*state + theta.dt*odeUpdateVector(theta)
    end
    return state[1:nSisters]
    end

    function transloglik(k::Int,xkm1::Union{Array{Int},Array{Float}},xk::Union{Array{Int},Array{Float}},P::Array{Float})
    whichstateprev = findfirst(w -> w>0, xkm1)
    whichstatenext = findfirst(w -> w>0 , xk)
    @assert !isnothing(whichstatenext) "oops: the next state is Nothing" ### since we had $xkm1 and then $xk" 
    transition_prob = P[whichstateprev,whichstatenext] #probability of getting between hidden states for xkm1 and xk   
    return log(transition_prob)
    end

    function approxtransmean(k::Int, xkm1::Union{AbstractArray{Int},AbstractArray{Float}}, ykm1::AbstractArray{Float}, yk::AbstractArray{Float}, theta::AbstractTheta,
                             u::Union{Nothing,AbstractArray{Float},Float}, P::AbstractArray{Float})
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev) "oops the previous state was $xkm1"
        transition_prob = P[whichstateprev,:]
        Q = Array{Float}(undef, length(transition_prob))
        copyto!(Q, log.(transition_prob)) #this needs to be reweighted appropriately due to conditioning
        for j=1:(nStates-1)
            xk = zeros(nStates)
            xk[j] = 1 #potential new state
            Q[j] += logpdf(MvNormal(ykm1 + theta.dt*odeUpdateMatrix(theta)*[ykm1; xk] + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),yk) #TODO could be more general for non isotropic gaussian
        end
        #anaphase state case separately
        Q[5] = logpdf(MvNormal(ykm1 + theta.dt.*[theta.v_ana; -theta.v_ana], sqrt(theta.dt/theta.tau)),yk) 
        b = maximum(Q)
        prob = exp.(Q .- b) #subtract for numerical stability
        prob ./= sum(prob) #normalise
        xk = stochasticTransition(prob,nStates,u)
        return xk
    end

    function approxloglik(k::Int, xkm1::Union{Array{Int},Array{Float}}, ykm1::Array{Float}, yk::Array{Float},
                          xk::Union{Array{Int},Array{Float}}, theta::AbstractTheta, P::Array{Float})
        whichstateprev = findfirst(w -> w>0, xkm1)
        whichstatenext = findfirst(w -> w>0 , xk)
        @assert !isnothing(whichstateprev)
        @assert !isnothing(whichstatenext)
        transition_prob = P[whichstateprev,:]
        Q = Array{Float}(undef, length(transition_prob))
        copyto!(Q, log.(transition_prob)) #this needs to be reweighted appropriately due to conditioning
        for j=1:(nStates-1)
            xk = zeros(nStates)
            xk[j] = 1 #potential new state
            Q[j] += logpdf(MvNormal(ykm1 + theta.dt*odeUpdateMatrix(theta)*[ykm1; xk] + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),yk) #TODO could be more general for non isotropic gaussian
        end
        #anaphase state case separately
        Q[5] = logpdf(MvNormal(ykm1 + theta.dt.*[theta.v_ana; -theta.v_ana], sqrt(theta.dt/theta.tau)),yk)
        b = maximum(Q)
        prob = exp.(Q .- b) #subtract for numerical stability
        prob ./= sum(prob) #normalise
        return log(prob[whichstatenext])
    end

    anaphasehmm = DiscreteState((k,xk,u=nothing) -> transmean(k,xk,u,P), (k,xk,ykm1) -> obsmean(k,xk,ykm1,th), R, convert(Int,nStates), convert(Int,nSisters))
    transll(k,xkm1,xk) = transloglik(k,xkm1,xk,P)
    approxll(k, xkm1, ykm1, yk, xk) = approxloglik(k, xkm1, ykm1, yk, xk, th, P)
    approxtrans(k, xkm1, ykm1, yk, u=nothing) = approxtransmean(k, xkm1, ykm1, yk, th, u, P)

    return anaphasehmm, transll, approxtrans, approxll 
end


