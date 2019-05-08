using Distributions

export
    theta,
    armondModel
    

#make the armond model as hmm class object thing

#set model parameters
struct theta
    tau::Float64
    alpha::Float64
    kappa::Float64
    v_minus::Float64
    v_plus::Float64
    k_icoh::Float64
    k_coh::Float64
    L::Float64
    dt::Float64
end

function armondModel(th=nothing)
th = (isnothing(th)) ? theta(450, 0.008, 0.025, -0.035, 0.015, 0.12/3.4, 0.06/3.4, 0.775, 3.4) : th #set default
nSisters = 2
angleTheta = 0 #for rotation when in 3D
tol = 10^(-3)

#define observation mean function
obsm(k,x) = x[1:nSisters] #observe the positions, but not the sigma variables
Q = th.dt/th.tau*eye(2) #variance matrix
R = tol*eye(2) #assume we observe exactly up to a small tolerance

    function odeUpdateMatrix(theta)
        M = [-theta.v_plus -theta.v_plus -theta.v_minus -theta.v_minus (-theta.kappa - theta.alpha) -theta.kappa; 
            theta.v_plus theta.v_minus theta.v_plus theta.v_minus theta.kappa (-theta.kappa - theta.alpha)]
        return M
    end

    function odeUpdateVector(theta)
        mu = [theta.kappa*theta.L*cos(angleTheta); -theta.kappa*theta.L*cos(angleTheta)]
        return mu
    end

    function fwdSimHiddenState(sigma::Array, theta)
    #state change stoich matrix
        S = [-1 -1 0 0 1 0 1 0;
              0 1 0 1 -1 -1 0 0;
              1 0 1 0 0 0 -1 -1;
              0 0 -1 -1 0 1 0 1]

    #hazard function (not as a function here though)
        h = [theta.k_icoh 0 0 0;
                           theta.k_icoh 0 0 0;
                           0 0 0 theta.k_icoh;
                           0 0 0 theta.k_icoh;
                           0 theta.k_coh 0 0;
                           0 theta.k_coh 0 0;
                           0 0 theta.k_coh 0;
                           0 0 theta.k_coh 0]

        prob = (ones(size(h)) .- exp.(-h.*theta.dt))*sigma
        @assert sum(prob) < 1 #model is not valid in this case
        cumProb = cumsum(prob, dims=1)
        u = rand(1)
        #decide which reaction to implement
        nReactions = size(S,2)
        for j = 1:nReactions
            if (u[1]<cumProb[j,1])
                sigma += S[:,j]
                break
            end
        end        
    return sigma
    end

    function fwdSimHiddenStateConditional(state::Array, obsk, theta)
    #added a column for dummy reaction of no change
    #state change stoich matrix
        S = [-1 -1 0 0 1 0 1 0 0;
              0 1 0 1 -1 -1 0 0 0;
              1 0 1 0 0 0 -1 -1 0;
              0 0 -1 -1 0 1 0 1 0]
    #hazard function (not as a function here though)
        h = [theta.k_icoh 0 0 0;
                           theta.k_icoh 0 0 0;
                           0 0 0 theta.k_icoh;
                           0 0 0 theta.k_icoh;
                           0 theta.k_coh 0 0;
                           0 theta.k_coh 0 0;
                           0 0 theta.k_coh 0;
                           0 0 theta.k_coh 0]
        prob = (ones(size(h)) .- exp.(-h.*theta.dt))*state[3:6]
        @assert sum(prob) < 1 #model is not valid in this case
        prob = [prob; 1-sum(prob)] #otherwise fire dummy reaction

    #weight the probability vector based on the observed data
        nReactions = size(S,2)
        weighting = zeros(nReactions)
        for j=1:nReactions
            nextstate = deepcopy(state)
            nextstate[3:6] += S[:,j] #what would the next hidden state be? Using fwd euler scheme gives all equal weights
            weighting[j] = pdf(MvNormal(nextstate[1:2] + theta.dt*odeUpdateMatrix(theta)*nextstate + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),obsk)
        end
#println(weighting)
        prob .*= weighting
        prob ./= sum(prob)
        cumProb = cumsum(prob, dims=1)
#println(cumProb)
        @assert isapprox(cumProb[nReactions], 1.0)
        u = rand(1)
        #decide which reaction to implement
        for j = 1:nReactions        
            if (u[1]<cumProb[j,1])
                state[3:6] += S[:,j]
                break
            end
        end
    return state
    end

    function transmean(k, state::Array, theta)    
    state[3:6] = fwdSimHiddenState(state[3:6], theta)
    #forward euler step
    state[1:2] += theta.dt*odeUpdateMatrix(theta)*state + theta.dt*odeUpdateVector(theta)
    return state
    end

    function transloglik(k,xkm1,xk,g,theta)
    #Quite specific to Armond model, could try to make more general
    #TODO: check normalizing constants in cases where we care about infering process/observation noise
    #hazard
    p_icoh = 1 - exp(-theta.k_icoh*theta.dt)
    p_coh = 1-exp(-theta.k_coh*theta.dt)
    P = [1-2*p_icoh p_icoh p_icoh 0;
        p_coh 1-2*p_coh 0 p_coh;
        p_coh 0 1-2*p_coh p_coh;
        0 p_icoh p_icoh 1-2*p_icoh]

    whichstateprev = findfirst(w -> w>0, xkm1[3:6])
    whichstatenext = findfirst(w -> w>0 , xk[3:6])
    @assert !isnothing(whichstatenext)
    transition_prob = P[whichstateprev,whichstatenext] #probability of getting between hidden states for xkm1 and xk   

    #where would the position have been before the brownian noise
    xkm1_updated = deepcopy(xkm1) #TODO check for similar copy errors
    xkm1_updated[3:6] = deepcopy(xk[3:6]) #based on the new states for sigma
    xkm1_updated[1:2] += theta.dt*odeUpdateMatrix(theta)*xkm1_updated + theta.dt*odeUpdateVector(theta)
    out = -norm(g.cholQ'\(xk[1:2] - xkm1_updated[1:2]))^2/2 + log(transition_prob)
    return out
    end

    function approxtransmean(k, xkm1, yk, theta)
    state = deepcopy(xkm1)
#println("xkm1: ", xkm1)
#println("yk: ", yk)
@assert !any([isnan(s) for s in state])
    state = fwdSimHiddenStateConditional(state, yk, theta)
    #forward euler step
    state[1:2] += theta.dt*odeUpdateMatrix(theta)*state + theta.dt*odeUpdateVector(theta)
#println("state: ", state)
    return state
    end

    function approxloglik(k, xkm1, yk, xk, g, theta)
    p_icoh = 1 - exp(-theta.k_icoh*theta.dt)
    p_coh = 1-exp(-theta.k_coh*theta.dt)
    P = [1-2*p_icoh p_icoh p_icoh 0;
        p_coh 1-2*p_coh 0 p_coh;
        p_coh 0 1-2*p_coh p_coh;
        0 p_icoh p_icoh 1-2*p_icoh]

    whichstateprev = findfirst(w -> w>0, xkm1[3:6])
    whichstatenext = findfirst(w -> w>0 , xk[3:6])
    @assert !isnothing(whichstatenext)
    xkm1_updated = deepcopy(xkm1) #TODO check for similar copy errors
    xkm1_updated[3:6] = deepcopy(xk[3:6]) #based on the new states for sigma
    Q = deepcopy(P[whichstateprev,:]) #this needs to be reweighted appropriately due to conditioning
    for j=1:4
    Q[j] *= pdf(MvNormal(xkm1[1:2] + theta.dt*odeUpdateMatrix(theta)*xkm1_updated + theta.dt*odeUpdateVector(theta),
                           sqrt(theta.dt/theta.tau)),yk)
    end
    Q ./= sum(Q) #normalise
    transition_prob = Q[whichstatenext] #probability of getting between hidden states for xkm1 and xk
    #where would the position have been before the brownian noise
    xkm1_updated[1:2] += theta.dt*odeUpdateMatrix(theta)*xkm1_updated + theta.dt*odeUpdateVector(theta)
    out = -norm(g.cholQ'\(xk[1:2] - xkm1_updated[1:2]))^2/2 + log(transition_prob)
    return out
    end

    armondhmm = NonLinearGaussian((k,xk) -> transmean(k,xk,th), obsm, Q, R, [1,2], [1,2], 6, 2)
    transll(k,xkm1,xk) = transloglik(k,xkm1,xk,armondhmm,th)
    approxll(k, xkm1, yk, xk) = approxloglik(k, xkm1, yk, xk, armondhmm, th)
    approxtrans(k, xkm1, yk) = approxtransmean(k, xkm1, yk, th)

    return armondhmm, transll, approxtrans, approxll 
end


