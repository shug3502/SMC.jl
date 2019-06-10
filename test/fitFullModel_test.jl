using SMC, Random, Distributions, DataFrames
trueValues = [450, 0.01, 0.025, -0.035, 0.015, 0.9, 0.95, 0.775]
thetaDF = DataFrame(paramNames = ["tau","alpha","kappa","v_minus","v_plus","p_icoh","p_coh","L"],
                    trueValues = trueValues)
th = thetaSimple(trueValues[1], trueValues[2], trueValues[3], trueValues[4],
    trueValues[5], trueValues[6], trueValues[7], trueValues[8], 2.0)
(armondhmmSimple, transll, approxtrans, approxll) = armondModelSimple(th)
hmm = HMM(armondhmmSimple, transll)
x0 = [0, 1, 0, 0];
y0 = [0.9, 0];

### generation from armond model
Random.seed!(234)
K=50
(states, observations) = generate(armondhmmSimple, x0, y0, K);

### preparation for plotting
using Gadfly, DataFrames
set_default_plot_size(21cm, 8cm)
dfObs = DataFrame(transpose(observations));
dfStates = DataFrame(transpose(states));
names!(dfObs, [Symbol("sister$i") for i in 1:2]) #change columns names, see https://stackoverflow.com/questions/21559683/how-do-you-change-multiple-column-names-in-a-julia-version-0-3-dataframe
names!(dfStates, [Symbol("++"), Symbol("+-"), Symbol("-+"), Symbol("--")])
dfT = DataFrame(time = th.dt*collect(1:K));
df = [dfT dfObs dfStates]
df = stack(df,[:sister1, :sister2], variable_name=Symbol("sisterID"),value_name=Symbol("position")) #reshape from wide to long
df = stack(df, 4:7, variable_name=Symbol("state"), value_name=Symbol("present"))
p1 = plot(df, x="time",y = "position", color="sisterID", Geom.line)
p2 = plot(df, x="time",y = "present", color="state", Geom.line)
hstack(p1,p2)

#prepare for parallel processing
using Distributed
addprocs(8)
module_dir = "/home/jonathanharrison/.julia/dev/SMC.jl"
@everywhere push!(LOAD_PATH, $module_dir) #see https://discourse.julialang.org/t/loading-modules-on-remote-workers-in-julia-1-0-7/13312/3
@everywhere using Distributions, SMC, Random

Random.seed!(125)
dimParams = 8
proposalWidthLU = [50 0 Inf;
                   0.01 0 Inf;
                   0.01 0 Inf;    
                   0.02 -Inf 0;     
                   0.02 0 Inf;           
                   0.05 0 1;
                   0.05 0 1;
                   0.05  0 Inf] #std, lower and upper for truncated normal proposal
paramProposal = [x -> TruncatedNormal(x,proposalWidthLU[i,1],proposalWidthLU[i,2],proposalWidthLU[i,3]) for i in 1:dimParams]
priors = [Gamma(1/0.5,1/0.001), #tau: shape vs rate paramterisation
    TruncatedNormal(0.01,sqrt(10000),0,Inf), #alpha
    TruncatedNormal(0.05,sqrt(10000),0,Inf), #kappa
    TruncatedNormal(-0.03,sqrt(10),-Inf,0), #v_minus
    TruncatedNormal(0.03, sqrt(10),0,Inf), #v_plus
    Beta(2,1), #p_icoh
    Beta(2.5,1), #p_coh
    TruncatedNormal(0.775,sqrt(0.121),0,Inf) #L
]
initialisationFn = [TruncatedNormal(450,100,0,Inf), #tau: shape vs rate paramterisation
    TruncatedNormal(0.01,0.01,0,Inf), #alpha
    TruncatedNormal(0.025,0.01,0,Inf), #kappa
    TruncatedNormal(-0.03,0.02,-Inf,0), #v_minus
    TruncatedNormal(0.03,0.02,0,Inf), #v_plus
    TruncatedNormal(0.9,0.1,0,1), #p_icoh
    TruncatedNormal(0.95,0.1,0,1), #p_coh
    TruncatedNormal(0.775,0.1,0,Inf) #L
]
if dimParams == 8
    subsetInd = 1:8
elseif dimParams == 2
    subsetInd = 6:7 #use simpler model with only p_coh and p_icoh
elseif dimParams == 3
    subsetInd = [2:3; 8] #with alpha, kappa and L
end
        
    proposalWidthLU = proposalWidthLU[subsetInd,:]
    priors = priors[subsetInd]
    initialisationFn = initialisationFn[subsetInd]

numIter = 20000
nChains = 6
N = 64;
numRandoms = K*(N+1);

varPilot = [1.25847e5 0.00259444 0.00378207 0.00879511 0.00355742 0.321018 0.166413 0.187392]
varPilot = varPilot[subsetInd]
scalingFactor = 2.56^(-1)/dimParams #see Golightly et al 2017
numIterPilot = 200
#set optimized Proposal
paramProposalOptim = [x -> TruncatedNormal(x,sqrt(varPilot[i])*scalingFactor,proposalWidthLU[i,2],proposalWidthLU[i,3]) for i in 1:dimParams];

pmcmcMethod = noisyMCMC #or use correlated
@time out = pmap(x -> pmcmcMethod(observations, priors,
         paramProposalOptim, dimParams, numRandoms,
         numIter=numIter, printFreq=1000, initialisationFn=initialisationFn,
         N=N, rho=0.99,resampler=sortedresample), 1:nChains);

#extracting from output of pmap is a bit messy; https://stackoverflow.com/questions/44234255/return-multiple-values-from-a-mapped-function
cTemp, actRate = map(x->getindex.(out, x), 1:2)
c =  zeros(numIter,dimParams,nChains)
for i=1:nChains
    c[:,:,i] = cTemp[i]
end

using MCMCChains, StatsPlots
theme(:ggplot2)
thetaDFCopy = thetaDF[subsetInd,:] #(dimParams==8) ? thetaDF : (dimParams==2) ? thetaDF[6:7,:] : nothing 
thin = 1
chn=Chains(c[1:thin:end,:,:], thetaDFCopy[:paramNames], start=1, thin=thin);
Rhat = gelmandiag(chn)
println(Rhat)

# visualize the MCMC simulation results
p1 = StatsPlots.plot(chn, colordim = :parameter);
p2 = StatsPlots.plot(chn, width=3, layout = [8,2]);
th_true = 0
for i=1:dimParams
    for j=1:dimParams
        if string(Rhat[:parameters][i]) == thetaDFCopy[:paramNames][j]
            th_true = thetaDFCopy[:trueValues][j]
            break
        end
    end
    StatsPlots.plot!(p2[i,1],1:numIter, repeat([th_true],numIter),color="black",linestyle=:dash)
#    StatsPlots.plot!(p2[i,2],repeat([th_true],100), range(0,stop=10/th_true,length=100),color="black")
end
display(p2)

