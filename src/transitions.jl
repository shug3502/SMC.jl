using SMCaux, LinearAlgebra, Distributions

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
#=
R1: 1 -> 3 or ++ -> -+
R2: 1 -> 2 or ++ -> +-
R3: 4 -> 3 or -- -> -+
R4: 4 -> 2 or -- -> +-
R5: 2 -> 1 or +- -> ++
R6: 2 -> 4 or +- -> --
R7: 3 -> 1 or -+ -> ++
R8: 3 -> 4 or -+ -> --
=#
p_icoh = 1 - exp(-theta.k_icoh*theta.dt)
p_coh = 1-exp(-theta.k_coh*theta.dt)
P = [1-2*p_icoh p_icoh p_icoh 0;
    p_coh 1-2*p_coh 0 p_coh;
    p_coh 0 1-2*p_coh p_coh;
    0 p_icoh p_icoh 1-2*p_icoh]

