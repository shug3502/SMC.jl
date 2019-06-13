using LinearAlgebra

export eye,
       nearestSPD

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
  elseif (r == 1) & (A <= 0)
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
  H = V*Sigma*V';
  # get Ahat in the above formula
  Ahat = (B+H)/2;
  # ensure symmetry
  Ahat = (Ahat + Ahat')/2;
  # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
  k = 0;
  while !isposdef(Ahat)
    k = k + 1;
      # Ahat failed the chol test. It must have been just a hair off,
      # due to floating point trash, so it is simplest now just to
      # tweak by adding a tiny multiple of an identity matrix.
    mineig = minimum(eigvals(Ahat));
    Ahat = Ahat + (-mineig*k.^2 + eps(mineig))*eye(size(A));
end

end
