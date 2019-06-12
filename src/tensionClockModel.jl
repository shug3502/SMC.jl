using Distributions

export
    thetaTensionClock,
    tensionClockModel

#set model parameters
struct thetaTensionClock <: AbstractTheta
    tau::Float
    alpha::Float
    kappa::Float
    v_minus::Float
    v_plus::Float
    L::Float
    r::Float
    rho::Float
    a_plus::Float
    a_minus::Float
    t_plus0::Float
    t_minus0::Float
    dt::Float
end

function tensionClockModel(th::Union{Nothing,thetaTensionClock}=nothing)
    th = (isnothing(th)) ? thetaTensionClock(450, 0.008, 0.025, -0.035, 0.015, 0.775,
                                             1.0, 2000, 3*10^(-4), 2.5*10^(-5), -0.1, 12.0, 2.0) : th #set default
    nSisters = convert(Int,2)
    nStates = convert(Int,4)
    angleTheta = convert(Float,0) #for rotation when in 3D
    R = th.dt/th.tau*eye(nSisters) #variance matrix
    time_since_switch = 0

    function computeStateMatrix(time_since_switch::Float, y::Array{Float})
        #matrix P must be calculated at each time step. This matrix gives probability of transition to each of 4 states
        T_plus = th.t_plus0 + th.a_plus*time_since_switch
        T_minus = th.t_minus0 - th.a_minus*time_since_switch
        T_delta = th.kappa*(y[1] - y[2] - th.L) #omit scaling by gamma compared to equations in draft manuscript
        p_plus = th.r/(1 + exp(th.rho*(T_delta - T_plus)))
        p_minus = th.r/(1 + exp(-th.rho*(T_delta - T_minus)))
        q_plus = 1 - p_plus
        q_minus = 1 - p_minus
        P = [q_plus*q_plus p_plus*q_plus p_plus*q_plus p_plus*p_plus;
             p_minus*q_plus q_minus*q_plus p_plus*p_minus p_plus*q_minus;
             p_minus*q_plus p_plus*p_minus q_minus*q_plus p_plus*q_minus;
             p_minus*p_minus p_minus*q_minus p_minus*q_minus q_minus*q_minus]
        return P
    end

    function transmean(k::Int, xkm1::Union{Array{Int},Array{Float}}, u::Union{Array{Float},Float,Nothing}, time_since_switch::Float)
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev)
        P = computeStateMatrix(time_since_switch, yk)
        prob = P[whichstateprev,:]
        xk = stochasticTransition(prob,nStates,u)
        @assert sum(xk) > 0 
        time_since_switch = (xk == xkm1) ? time_since_switch + th.dt : 0
        return (xk, time_since_switch)
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

    function transloglik(k::Int,xkm1::Union{Array{Int},Array{Float}},xk::Union{Array{Int},Array{Float}})
    whichstateprev = findfirst(w -> w>0, xkm1)
    whichstatenext = findfirst(w -> w>0 , xk)
    @assert !isnothing(whichstatenext) "oops: the next state is Nothing" ### since we had $xkm1 and then $xk" 
    P = computeStateMatrix(time_since_switch, yk)
    transition_prob = P[whichstateprev,whichstatenext] #probability of getting between hidden states for xkm1 and xk   
    return log(transition_prob)
    end

    function approxtransmean(k::Int, xkm1::Union{AbstractArray{Int},AbstractArray{Float}}, ykm1::AbstractArray{Float}, yk::AbstractArray{Float}, theta::thetaSimple,
                             u::Union{Nothing,AbstractArray{Float},Float})
        whichstateprev = findfirst(w -> w>0, xkm1)
        @assert !isnothing(whichstateprev) "oops the previous state was $xkm1"
        P = computeStateMatrix(time_since_switch, yk)
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
                          xk::Union{Array{Int},Array{Float}}, theta::thetaSimple)
        whichstateprev = findfirst(w -> w>0, xkm1)
        whichstatenext = findfirst(w -> w>0 , xk)
        @assert !isnothing(whichstateprev)
        @assert !isnothing(whichstatenext)
        P = computeStateMatrix(time_since_switch, yk)
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

    tensionClockhmm = DiscreteState((k,xk,u=nothing) -> transmean(k,xk,u,P), (k,xk,ykm1) -> obsmean(k,xk,ykm1,th), R, convert(Int,4), convert(Int,2))
    transll(k,xkm1,xk) = transloglik(k,xkm1,xk,P)
    approxll(k, xkm1, ykm1, yk, xk) = approxloglik(k, xkm1, ykm1, yk, xk, th, P)
    approxtrans(k, xkm1, ykm1, yk, u=nothing) = approxtransmean(k, xkm1, ykm1, yk, th, u, P)

    return tensionCloackhmm, transll, approxtrans, approxll 
end


