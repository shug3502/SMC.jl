using LinearAlgebra, DataFrames, Gadfly

export eye,
       nearestSPD,
       extractOutput,
       proportion_of_js,
       plotHiddenStates

function eye(m::Integer)
    #shortcut to avoid lots of renaming
    out = convert(Array{Float,2},Matrix(1.0I, m, m))
return out
end

#################################################

function nearestSPD(A)
  #=
  nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
   usage: Ahat = nearestSPD(A)
  
   From Higham: "The nearest symmetric positive semidefinite matrix in the
   Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
   where H is the symmetric polar factor of B=(A + A')/2."
  
   http://www.sciencedirect.com/science/article/pii/0024379588902236
  
   arguments: (input)
    A - square matrix, which will be converted to the nearest Symmetric
      Positive Definite Matrix.

   Arguments: (output)
    Ahat - The matrix chosen as the nearest SPD matrix to A.
  =#
  ###############################################
  
  # test for a square matrix A
  r,c = size(A);
  if r != c
    error("A must be a square matrix.")
  elseif (r == 1) & any(A .<= 0)
    # A was scalar and non-positive, so just return eps
    Ahat = eps;
    return Ahat
  end

  # symmetrize A into B
  B = (A + A')/2;
  # Compute the symmetric polar factor of B. Call it H.
  # Clearly H is itself SPD.
  F = svd(B);
  U, Sigma, V = F.U, F.S, F.V;
  H = V*diagm(0 => Sigma)*V';
  # get Ahat in the above formula
  Ahat = (B+H)/2;
  # ensure symmetry
  Ahat = (Ahat + Ahat')/2;
  # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
  k = 0;
  while !isposdef(Ahat)
    k += 1;
      # Ahat failed the chol test. It must have been just a hair off,
      # due to floating point trash, so it is simplest now just to
      # tweak by adding a tiny multiple of an identity matrix.
    mineig = minimum(eigvals(Ahat));
    Ahat = Ahat + (-mineig*k.^2 + eps(mineig))*eye(size(A,1));
  end
return Ahat
end
####################

function extractOutput(out::Array; dimParams::Int=8,nChains::Int=6,K::Int=100)
    #extracting from output of pmap is a bit messy; https://stackoverflow.com/questions/44234255/return-multiple-values-from-a-mapped-function
    cTemp, hiddenStatesTemp, actRate = map(x->getindex.(out, x), 1:3)
    nIter_minusburnin = size(cTemp[1],1)
    c =  zeros(nIter_minusburnin,dimParams,nChains)
    hiddenStates = zeros(Int, nIter_minusburnin*nChains, K) #combine chains for hidden states
    for i=1:nChains
        c[:,:,i] = cTemp[i]
        hiddenStates[range((i-1)*nIter_minusburnin+1,stop=i*nIter_minusburnin,step=1),:] = hiddenStatesTemp[i]
    end
    return c, hiddenStates
end

####################

# count proportion of MCMC iterations for each hidden state at given time and plot
function proportion_of_js(states, j::Int)
sum(states .== j)/length(states)
end

#####################

function plotHiddenStates(states::Array; dt::Float=2.0)
    dimx = size(states,1)
    K = size(states,2)
    dfStates = DataFrame(transpose(states));
    stateNames = [Symbol("++"), Symbol("+-"), Symbol("-+"), Symbol("--"),Symbol("Anaphase")]
    names!(dfStates, stateNames[1:dimx])
    dfT = DataFrame(time = dt*collect(1:K));
    df = [dfT dfStates]
    df = stack(df, 2:(dimx+1), variable_name=Symbol("state"), value_name=Symbol("present"))
    p = Gadfly.plot(df, x="time",y = "present", color="state", Geom.line)
    return p
end

#####################

function plotObsAndStates(observations::Array,states::Array; dt::Float=2.0)
    dimx = size(states,1)
    dfObs = DataFrame(transpose(observations));
    dfStates = DataFrame(transpose(states));
    names!(dfObs, [Symbol("sister$i") for i in 1:2]) #change columns names, see https://stackoverflow.com/questions/21559683/how-do-you-change-multiple-column-names-in-a-julia-version-0-3-dataframe
    stateNames = [Symbol("++"), Symbol("+-"), Symbol("-+"), Symbol("--"),Symbol("Anaphase")]
    names!(dfStates, stateNames[1:dimx])
    dfT = DataFrame(time = dt*collect(1:K));
    df = [dfT dfObs dfStates]
    df = stack(df,[:sister1, :sister2], variable_name=Symbol("sisterID"),value_name=Symbol("position")) #reshape from wide to long
    df = stack(df, 4:(3+hmm.dimx), variable_name=Symbol("state"), value_name=Symbol("present"))
    p1 = plot(df, x="time",y = "position", color="sisterID", Geom.line)
    p2 = plot(df, x="time",y = "present", color="state", Geom.line)
    return hstack(p1,p2)
end

