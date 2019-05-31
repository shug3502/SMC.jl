export eye

#shortcut to avoid lots of renaming?

function eye(m::Integer)
    out = convert(Array{Float,2},Matrix(1.0I, m, m))
return out
end

