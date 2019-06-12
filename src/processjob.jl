using MAT, DataFrames

export processjob

function processjob(path=nothing)
if isnothing(path)
path = "/home/jonathanharrison/kittracking001-kitjobset_190509_JHv203_anaphase-capture_3_cropped.ome.mat"
end
job = matread(path);

coords1 = job["dataStruct"][1]["sisterList"]["coords1"]
coords2 = job["dataStruct"][1]["sisterList"]["coords2"]
nSisters = length(coords1)
K = size(coords1[1],1)

outputFormat = zeros(2*K*nSisters,9)
for j=1:nSisters
    additionalStuff = hcat(1:K,repeat([j],K),repeat([1],K))
    outputFormat[range((j-1)*K+1,stop=j*K,step=1),:] = hcat(job["dataStruct"][1]["sisterList"]["coords1"][j], additionalStuff)
    additionalStuff = hcat(1:K,repeat([j],K),repeat([2],K))
    outputFormat[range(K*nSisters+(j-1)*K+1,stop=K*nSisters+j*K,step=1),:] = hcat(job["dataStruct"][1]["sisterList"]["coords2"][j], additionalStuff)
end

df = DataFrame(outputFormat)
names!(df,[Symbol("Position_1"),Symbol("Position_2"),Symbol("Position_3"),
        Symbol("Amplitude_1"),Symbol("Amplitude_2"),Symbol("Amplitude_3"),
        Symbol("Frame"),Symbol("SisterPairID"),Symbol("SisterID")])
   
return df 
end
