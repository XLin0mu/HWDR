#
#Author: Xlin0mu
#Date: 2022-10-22 23:55
#LastEditTime: 2022-10-23 02:07
#Description: Description
#Edited by TAMRIKer
#Copyright (c) 2022 by Xlin0mu, All Rights Reserved. 
#


#region_Universal

module Universal_Xlin0mu
#This module is created to supply some funcitoin which are very universal
export oorand,reLU,sigmoid,donothing,softmax

"""creat a rand number from -1 to 1"""
function oorand()
    return 2 * (rand() - 0.5)
end

"""usage is similar with rand(T, dims...)"""
function oorand(T, s...)
    a = Array{T}(undef, s...)
    for i in 1 : lastindex(a)
        a[i] = oorand()
    end
    return a
end

"""
The argument d decide what's your input\n
0 means input a "x"
dx means input a "x", but calculate its derivative
dy means input a "y", but calculate its derivative
"""
function reLU(arg::Real; d = 0)
    if d in Set((0,"dx","dy"))
        if arg <= 0
            return zero(arg)
        else
            if d == 0
                return arg
            else
                return one(arg)
            end
        end
    end
    return null
end

"""
The argument d decide what's your input\n
0 means input a "x"
dx means input a "x", but calculate its derivative
dy means input a "y", but calculate its derivative
"""
function sigmoid(arg::Real; d = 0)
    if d in Set((0,"dx","dy"))
        if d == "dy"
            return arg*(1-arg)
        else snum = 1 / (1 + ℯ^(-arg))
            if d == 0
                return snum
            else
                return snum*(1-snum)
            end
        end
    end
    return null
end

function softmax(a :: Vector; d = 0)
    if d in Set((0, "dx"))
        aa = exp.(a)
        if d == 0
            return aa ./= sum(aa)
        else
            s = sum(aa)
            return aa.*(s.-aa) ./ s^2
        end
    end
    return nothing
end

function donothing(a)
    return a
end

end

#endregion_Universal
#region_ANN

module ANN_Xlin0mu

using Main.Universal_Xlin0mu
export ANNLayer,ANN
export runANN, trainANN!, testANN

#region_ANN_structs

struct ANNLayer
    iostruct    :: Tuple
    bias        :: Vector{Float64}
    weights     :: Matrix{Float64}
    disposion   :: Function
end

function ANNLayer(i, o, disposion; creater :: Function = oorand)
    ANNLayer((i, o), creater(Float64, o), creater(Float64, o, i), disposion)
end

struct ANN
    layers      :: Vector{ANNLayer}
    effector    :: Function
end

struct ANNTestdata
    data    :: Vector
    axons   :: Vector
end

struct ANNBPdata
    layers_bias     :: Vector
    layers_weights  :: Vector
    function ANNBPdata(lb, lw)
        if length(lb) != length(lw)
            throw(ArgumentError)
        end
        new(lb, lw)
    end
end

function Base.:+(bp1 :: ANNBPdata, bp2 :: ANNBPdata)
    bp = ANNBPdata(
    bp1.layers_bias     +  bp2.layers_bias,
    bp1.layers_weights  +  bp2.layers_weights
    )
    return bp
end

function Base.:*(bp1 :: ANNBPdata, c :: Real)
    bp = ANNBPdata(
    bp1.layers_bias     * c,
    bp1.layers_weights  * c
    )
    return bp
end

function Base.:+(ann :: ANN, bp :: ANNBPdata)
    newann = ann
    if length(ann.layers) != length(bp.layers_bias)
        throw(ArgumentError("ann + bp doesn't evaluable"))
    end
    for l in 1 : length(ann.layers)
        newann.layers[l].bias      .+=  bp.layers_bias[l]
        newann.layers[l].weights   .+=  bp.layers_weights[l]
    end
    return newann
end
#endregion_ANN_structs
#region_ANN_functions

"""run a layer with giving data, return its output"""
function runLayer(layer :: ANNLayer, data :: Vector)
    if layer.iostruct[1] != length(data)
        throw(ArgumentError("layer's input requires dims as $(layer.iostruct[1]), but putin actually get $(size(data))" ))
    end
    return layer.disposion.(layer.weights * data + layer.bias)
end

"""run a ANN by traverse its all layer, and give its output by ann's effector"""
function runANN(ann :: ANN, data :: Vector; test = false)
    axons = Vector{Vector{Any}}(undef, length(ann.layers))
    axons[1] = runLayer(ann.layers[1], data)
    for lay in 2 : length(ann.layers)
        axons[lay] = runLayer(ann.layers[lay], axons[lay - 1])
    end
    if test == false
        return eff = ann.effector(axons[end])
    elseif test == true
        return ANNTestdata(data, axons)
    end
end

"""calculate average variance"""
function calcost(ann :: ANN, td :: ANNTestdata, expect :: Any)
    return sum((td.axons[end] .- ann.effector(td.axons[end])).^2)/ann.layers[end].iostruct[2]
end

"""calculate ann's backward prapogation by give data"""
function calBP(ann :: ANN, td :: ANNTestdata, label :: Any)
    lays = length(ann.layers)
    lb = Vector{Vector{Float64}}(undef, lays)
    lw = Vector{Matrix{Float64}}(undef, lays)
    #lb[end] = ann.effector(ann, td, label).* ann.layers[end].disposion.(td.axons[end]; d = "dy")
    lb[end] = ann.effector(ann, td, label).* ann.layers[end].disposion.(td.axons[end]; d = "dy")
    if lays != 1
        for l in lays :-1: 2
            #lw[l] = lb[l] * td.axons[l-1]'
            lw[l] = lb[l] .* (ones(Float64, length(lb[l])) * td.axons[l-1]')
            #lb[l-1] = (ann.layers[l].weights .* lb[l])' * ones(ann.layers[l].iostruct[2]) .* ann.layers[l-1].disposion.(td.axons[l-1]; d = "dy") 
            lb[l-1] = ann.layers[l-1].disposion.(td.axons[l-1]; d = "dx") .* (lb[l] .* ann.layers[l].weights)' * ones(Float64, length(lb[l]))
        end
    end
    #lw[1] = lb[1] * td.data'
    lw[1] = lb[1] * td.data'
    throw(ArgumentError("CNM"))
    return ANNBPdata(lb, lw)
end

"""calculate an 's backward prapogation arguments by giving data set"""
function getBP(ann :: ANN, data :: Vector{Vector{Float64}}, labels :: Vector)
    h = length(data)
    bp =  calBP(ann, runANN(ann, data[1]; test = true), labels[1])
    for i in 2 : h
        bp += calBP(ann, runANN(ann, data[i]; test = true), labels[i])
    end
    return bp * h^-1
end

"""
choose a mode as follows\n
fullbatch\n
minibatch(circle = n, step = num)\n
Waring!(False result)\n
random-descent\n
"""
function trainANN!(ann :: ANN, data :: Vector{Vector{Float64}}, labels :: Vector; mode = "fullbatch", mode_arg = nothing, rate = 1)
    if mode == "fullbatch"
        bp = getBP(ann, data, labels)
        ann += bp * rate
    end
    if mode == "minibatch"
        l = length(data)
        step = mode_arg.step
        st = 1
        n = 0
        while n < mode_arg.circle
            for i in st : step : l
                minidata = data[i:i + st - 1]
                bp = getBP(ann, minidata, labels)
                ann += bp * rate
            end
            if l%step != 0
                st = step + 1 - l%step
                minidata = data[i : l; 1 : st-1]
                bp = getBP(ann, minidata, labels)
                ann += bp * rate
            end
            n+=1
        end
    end
    if mode == "random-descent"
        for i in 1 : length(data)
            ann += getBP(ann, [data[i]], [labels[i]]) * rate
        end
    end
    return nothing
end

function testANN(ann :: ANN, data :: Vector, lables :: Vector)
    l = length(data)
    if l != length(lables)
        throw(ArgumentError)
    end
    c = n = 0
    for i in 1 : l
        td = runANN(ann, data[i]; test = true)
        if ann.effector(td.axons[end]) == lables[i]
            n+=1
        end
        c += calcost(ann, td, lables[i])
    end
    println("the accuracy of this ANN calculated with giving data is\n$(100n/length(data))%")
    println("the avrage cost of this ANN calculated with giving data is\n", c/length(data))
    return nothing
end

#endregion_ANN_functions
#region_ANN_hwdr
function effector_hwdr(axons_o::Vector{Any})
    return findmax(softmax(axons_o))[2] - 1
end

function effector_hwdr(ann :: ANN, td :: ANNTestdata, num :: Integer)
    expect = zeros(ann.layers[end].iostruct[2])[num+1] = 1
    return -2 .* (expect .- softmax(td.axons[end])) .* softmax(td.axons[end]; d = "dx")
end

end
#endregion_ANN_hwdr

#endregion_ANN

using Main.Universal_Xlin0mu
using Main.ANN_Xlin0mu
import Main.ANN_Xlin0mu.effector_hwdr

hwdr_newANN = ANN([
    ANNLayer(784, 32, reLU),
    ANNLayer(32, 32, reLU),
    ANNLayer(32, 10, sigmoid)],
    effector_hwdr
)

using JLD2

data = load("MNISTdata.jld2")
#train_tensor include 15000 data
#train_lables include 15000 data
#test_tensor include 3500 data
#test_labels include 3500 data

testANN(hwdr_newANN, data["test_tensor"], data["test_labels"])
for i in 1:1
    trainANN!(hwdr_newANN, data["train_tensor"], data["train_labels"]; mode = "mini-batch", mode_arg = (1, 300), rate = 0.01)
end
testANN(hwdr_newANN, data["test_tensor"], data["test_labels"])


