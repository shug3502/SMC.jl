#!/usr/bin/env julia

using CUDAdrv, CUDAnative

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const M = typemax(Int32)
const A = Int32(1103515245)
const C = Int32(12345)
const threads_per_block = 512


# Utility functions

function rounddouble(value)
    new_value = convert(Int, floor(value))
    if (value - new_value < 0.5)
        return new_value
    else
        # NOTE: this is wrong, but we mimic the behavior
        return new_value
    end
end

function randu(seed, index)
    num::Int32 = A * seed[index] + C
    seed[index] = num % M
    q = seed[index]/M
    return abs(q)
end

function randn(seed, index)
    u = randu(seed, index)
    v = randu(seed, index)
    cosine = cos(2 * pi * v)
    rt = -2 * log(u)
    return sqrt(rt) * cosine
end


# Particle filter
# Helper functions

@inline function cdf_calc(
    CDF,        # Out
    weights,    # Int
    Nparticles)
    CDF[1] = weights[1]
    for x=2:Nparticles
        CDF[x] = weights[x] + CDF[x-1]
    end
end

@inline function d_randu(seed, index)
    num = A * seed[index] + C
    seed[index] = num % M
    return CUDAnative.abs(seed[index]/M)
end

@inline function d_randn(seed, index)
    u = d_randu(seed, index)
    v = d_randu(seed, index)
    cosine = CUDAnative.cos(2*pi*v)
    rt = -2 * CUDAnative.log(u)
    return CUDAnative.sqrt(rt) * cosine
end

@inline function calc_likelihood_sum(I, ind, num_ones, index)
    likelihood_sum = Float64(0)
    for x=1:num_ones
        i = ind[(index-1) * num_ones + x]
        v = ((I[i] - 100)*(I[i] - 100)
            - (I[i] -228)*(I[i] -228))/50
        likelihood_sum += v
    end
    return likelihood_sum
end

@inline function dev_round_double(value)::Int32
    new_value = unsafe_trunc(Int32, value)
    if value - new_value < .5f0
        return new_value
    else
        # NOTE: keep buggy semantics of original, should be new_value+1
        return new_value
    end
end


# Kernels

function find_index_kernel(arrayX, arrayY, CDF, u, xj, yj, weights, Nparticles)
    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

    if i <= Nparticles
        index = 0   # an invalid index
        for x=1:Nparticles
            if CDF[x] >= u[i]
                index = x
                break
            end
        end
        if index == 0
            index = Nparticles
        end

        xj[i] = arrayX[index]
        yj[i] = arrayY[index]
    end
    sync_threads()
    return
end

function normalize_weights_kernel(weights, Nparticles, partial_sums, CDF, u, seed)
    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

    shared = @cuStaticSharedMem(Float64, 2)
    u1_i = 1
    sum_weights_i = 2
    # shared[1] == u1, shared[2] = sum_weights

    if threadIdx().x == 1
        shared[sum_weights_i] = partial_sums[1]
    end
    sync_threads()

    if i <= Nparticles
        weights[i] = weights[i] / shared[sum_weights_i]
    end
    sync_threads()

    if i==1
        cdf_calc(CDF, weights, Nparticles)
        u[1] = (1/Nparticles) * d_randu(seed, i)
    end
    sync_threads()

    if threadIdx().x == 1
        shared[u1_i] = u[1]
    end
    sync_threads()

    if i <= Nparticles
        u1 = shared[u1_i]
        u[i] = u1 + i / Nparticles
    end
    return
end

function sum_kernel(partial_sums, Nparticles)
    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

    if i==1
        sum = 0.0
        num_blocks = unsafe_trunc(Int,CUDAnative.ceil(Nparticles/threads_per_block))
        for x=1:num_blocks
            sum += partial_sums[x]
        end
        partial_sums[1] = sum
    end
    return
end

function likelihood_kernel(array, j, ind, objxy, likelihood, I, weights,
                           count_ones, k, IszY, Nfr, partial_sums, param)
    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

    buffer = @cuStaticSharedMem(Float64, 512)
    if i <= param.Nparticles
        array.X[i] = j.x[i]
        array.Y[i] = j.y[i]
        weights[i] = 1/param.Nparticles

        array.X[i] = array.X[i] + 1.0 + 5.0 * d_randn(param.seed, i)
        array.Y[i] = array.Y[i] - 2.0 + 2.0 * d_randn(param.seed, i)
    end

    sync_threads()

    if i <= param.Nparticles
        for y=0:count_ones-1
            indX = dev_round_double(array.X[i]) + objxy[y*2 + 2]
            indY = dev_round_double(array.Y[i]) + objxy[y*2 + 1]

            ind[(i-1)*count_ones + y + 1] = CUDAnative.abs(indX*IszY*Nfr + indY*Nfr + k - 1) + 1
            if ind[(i-1)*count_ones + y + 1] > param.max_size
                ind[(i-1)*count_ones + y + 1] = 1
            end
        end
        likelihood[i] = calc_likelihood_sum(I, ind, count_ones, i)
        likelihood[i] = likelihood[i]/count_ones
        weights[i] = weights[i] * CUDAnative.exp(likelihood[i])
    end
    buffer[threadIdx().x] = 0.0

    sync_threads()

    if i <= param.Nparticles
        buffer[threadIdx().x] = weights[i]
    end
    sync_threads()

    s = div(blockDim().x,2)
    while s > 0
        if threadIdx().x <= s
            v = buffer[threadIdx().x]
            v += buffer[threadIdx().x + s]
            buffer[threadIdx().x] = v
        end
        sync_threads()
        s>>=1
    end
    if threadIdx().x == 1
        partial_sums[blockIdx().x] = buffer[1]
    end
    sync_threads()
    return
end

function getneighbors(se::Array{Int}, num_ones, neighbors::Array{Int}, radius)
    neighY = 1
    center = radius -1
    diameter = radius * 2 -1
    for x=0:diameter-1
        for y=0:diameter-1
            if se[x*diameter + y + 1] != 0
                neighbors[neighY * 2 - 1] = y - center
                neighbors[neighY * 2] = x - center
                neighY += 1
            end
        end
    end
end

function particlefilter(I::Array{UInt8}, IszX, IszY, Nfr, seed::Array{Int32}, Nparticles)
    max_size = IszX * IszY * Nfr

    # Original particle centroid
    xe = rounddouble(IszY/2.0)
    ye = rounddouble(IszX/2.0)

    # Expected object locations, compared to cneter
    radius = 5
    diameter = radius * 2 -1
    disk = Vector{Int}(undef, diameter * diameter)
    streldisk(disk, radius)
    count_ones = 0
    for x=1:diameter
        for y=1:diameter
            if disk[(x-1) * diameter + y] == 1
                count_ones += 1
            end
        end
    end

    objxy = Vector{Int}(undef, count_ones * 2)
    getneighbors(disk, count_ones, objxy, radius)

    # Initial weights are all equal (1/Nparticles)
    weights = Vector{Float64}(undef, Nparticles)
    for x=1:Nparticles
        weights[x] = 1 / Nparticles
    end

    # Initial likelihood to 0.0
    g_likelihood = CuArray{Float64}(Nparticles)
    g_arrayX = CuArray{Float64}(Nparticles)
    g_arrayY = CuArray{Float64}(Nparticles)
    xj = Vector{Float64}(undef, Nparticles)
    yj = Vector{Float64}(undef, Nparticles)
    g_CDF = CuArray{Float64}(Nparticles)

    g_ind = CuArray{Int}(count_ones * Nparticles)
    g_u = CuArray{Float64}(Nparticles)
    g_partial_sums = CuArray{Float64}(Nparticles)

    for x=1:Nparticles
        xj[x] = xe
        yj[x] = ye
    end

    num_blocks = Int(ceil(Nparticles/threads_per_block))

    g_xj = CuArray(xj)
    g_yj = CuArray(yj)
    g_objxy = CuArray(objxy)
    g_I = CuArray(I)
    g_weights = CuArray(weights)
    g_seed = CuArray(seed)

    for k=2:Nfr
        @cuda blocks=num_blocks threads=threads_per_block likelihood_kernel(
            (X=g_arrayX, Y=g_arrayY), (x=g_xj, y=g_yj), g_ind,
            g_objxy, g_likelihood, g_I, g_weights,
            count_ones, k, IszY, Nfr, g_partial_sums,
            (max_size=max_size, Nparticles=Nparticles, seed=g_seed))

        @cuda blocks=num_blocks threads=threads_per_block sum_kernel(
            g_partial_sums, Nparticles)

        @cuda blocks=num_blocks threads=threads_per_block normalize_weights_kernel(
            g_weights, Nparticles, g_partial_sums, g_CDF, g_u, g_seed)

        @cuda blocks=num_blocks threads=threads_per_block find_index_kernel(
            g_arrayX, g_arrayY, g_CDF, g_u, g_xj, g_yj, g_weights, Nparticles)
    end

    arrayX = Array(g_arrayX)
    arrayY = Array(g_arrayY)
    weights = Array(g_weights)

    xe = ye = 0
    for x=1:Nparticles
        xe += arrayX[x] * weights[x]
        ye += arrayY[x] * weights[x]
    end

    if OUTPUT
        outf = open("output.txt", "w")
    else
        outf = stdout
    end
    println(outf,"XE: $xe")
    println(outf,"YE: $ye")
    distance = sqrt((xe - Int(rounddouble(IszX/2.0)))^2
                   +(ye - Int(rounddouble(IszY/2.0)))^2)
    println(outf,"distance: $distance")

    if OUTPUT
      close(outf)
    end

end

