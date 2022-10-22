#
#Author: Xlin0mu
#Date: 2022-10-22 23:55
#LastEditTime: 2022-10-23 02:07
#Description: Description
#Edited by TAMRIKer
#Copyright (c) 2022 by Xlin0mu, All Rights Reserved. 
#


#region_predefine

function reLU(arg::Real)
    if arg < 0
        return zero(arg)
    else
        return arg
    end
end

function dreLU(num::Number)
    if max(0, num) > 0
        return one(num)
    else
        return zero(num)
    end
end

function sigmoid(arg::Real)
    return 1 / (1 + ℯ^(-arg))
end

function dsigmoid(num::Real)
    #snum = sigmoid(num)
    #return snum * (1 - snum)
    return num * (1 - num)
end

function donothing(a)
    return a
end

"""creat a rand number from -1 to 1"""
function myrand()
    return 2 * (rand() - 0.5)
end

"""usage is similar with rand(T, dims...)"""
function myrand(T, s...)
    a = Array{T}(undef, s...)
    for i in 1 : lastindex(a)
        a[i] = myrand()
    end
    return a
end

"""
I use this to prevent my Backward Propagation to be too negative
"A baby needs praise, so a ANN too" -by nobody
"""
function appreciate!(n :: Real, applevel)
    if n >=0
        return (applevel+1)*n
    end
    return n
end

function appreciate!(some :: Array, apl)
    return broadcast(appreciate!, some, apl)
end

#endregion_predefine
#region_structs

"""
Neuron's dendrites mean weights' array, they work just like the synaptic receptors in dendrites
"""
mutable struct Neuron
    dendrites::Array
    bias::Float64
    axonfunction::Function
    Neuron(a, b, c=reLU) = new(a, b, c)
end

struct ANN
    neulay_i::Vector{Neuron}
    neulay_mid::Vector{Vector{Neuron}}
    neulay_o::Vector{Neuron}
    effector::Function
    pretreatment::Function
end

function ANN(neulay_i, neulay_mid::Array{Neuron}, neulay_o, effector; pretreatment=donothing)
    return ANN(neulay_i, (neulay_mid,), neulay_o, effector, pretreatment)
end

function ANN(neulay_i, neulay_o, effector, neulay_mid...; pretreatment=donothing)
    return ANN(neulay_i, neulay_mid, neulay_o, effector, pretreatment)
end

"""to create a standard ANN which full of zeros"""
function creatANN(receptor::Tuple, input::Integer, middle::Tuple, output::Integer, effector::Function, pretreat::Function=donothing)
    #create ann with undef data
    i = Vector{Neuron}(undef, input)
    o = Vector{Neuron}(undef, output)
    m = Vector{Vector{Neuron}}(undef, length(middle))
    for j in 1:length(middle)
        m[j] = Vector{Neuron}(undef, middle[j])
    end
    ann = ANN(i, m, o, effector, pretreat)

    #set the precise struct size
    for j in 1:input
        i[j] = Neuron(myrand(Float64, receptor...), myrand())
    end
    for j in 1:middle[1]
        m[1][j] = Neuron(myrand(Float64, input), myrand())
    end
    if length(middle) != 1
        for l in length(middle):2
            for j in 1:middle[l]
                m[l][j] = Neuron(myrand(Float64, middle[l-1]), myrand())
            end
        end
    end
    for j in 1:output
        o[j] = Neuron(myrand(Float64, middle[lastindex(middle)]), myrand(), sigmoid)
    end
    return ann
end

struct ANNTestdata
    stimuli::Array{Float64}
    axons_i::Vector{Float64}
    axons_mid::Vector{Vector{Float64}}
    axons_o::Vector{Float64}
    expvalue::Float64
end

mutable struct ANNBPargs
    neulay_i_b::Vector{Float64}
    neulay_i_dd::Vector{Array{Float64}}
    neulay_o_b::Vector{Float64}
    neulay_o_dd::Vector{Array{Float64}}
    neulay_mid_b::Vector{Vector{Float64}}
    neulay_mid_dd::Vector{Vector{Vector{Float64}}}
end

"""using to initialize the args variety"""
function ANNBPargs(ann::ANN)
    n_i_b = zeros(length(ann.neulay_i))
    n_i_dd = fill!(Vector{Array{Float64}}(undef, length(ann.neulay_i)), zeros(Float64, size(ann.neulay_i[1].dendrites)...))
    n_o_b = zeros(length(ann.neulay_o))
    n_o_dd = fill!(Vector{Array{Float64}}(undef, length(ann.neulay_o)), zeros(Float64, size(ann.neulay_o[1].dendrites)...))
    n_mid_b = Vector{Vector{Float64}}(undef, length(ann.neulay_mid))
    for l in 1:length(n_mid_b)
        n_mid_b[l] = zeros(length(ann.neulay_mid[l]))
    end
    n_mid_dd = Vector{Vector{Vector{Float64}}}(undef, length(ann.neulay_mid))
    for l in 1:length(n_mid_dd)
        n_mid_dd[l] = Vector{Vector{Float64}}(undef, length(ann.neulay_mid[l]))
        for i in 1:length(n_mid_dd[l])
            n_mid_dd[l][i] = zeros(size(ann.neulay_mid[l][1].dendrites))
        end
    end
    return ANNBPargs(n_i_b, n_i_dd, n_o_b, n_o_dd, n_mid_b, n_mid_dd)
end

#endregion_structs
#region_general_funcitons

function effector_hwdr(axons_o::Vector{Float64})
    return findmax(axons_o)[2][1] - 1
end

"""Recept a input, calcul it by neuron, return a value which equal to this neuron's axon value """
function sense(recept, neuron::Neuron)
    if size(recept) != size(neuron.dendrites)
        throw(ArgumentError, "incorrect input, the recept's size should be similar with neuron.dendrites's")
    end
    return neuron.axonfunction(sum(neuron.dendrites .* recept) + neuron.bias)
end

function reflectANN(stimuli, ann::ANN; test=false)
    #define some varies for later dealing
    axons_i = zeros(Float64, length(ann.neulay_i))
    axons_o = zeros(Float64, length(ann.neulay_o))
    axons_mid = Vector(undef, length(ann.neulay_mid))
    for l in 1:length(ann.neulay_mid)
        axons_mid[l] = zeros(Float64, length(ann.neulay_mid[l]))
    end

    #strat to sense
    for i in 1:length(ann.neulay_i)
        axons_i[i] = sense(stimuli, ann.neulay_i[i])
    end

    for i in 1:length(ann.neulay_mid[1])
        axons_mid[1][i] = sense(axons_i, ann.neulay_mid[1][i])
    end
    if length(ann.neulay_mid) != 1
        for l in 2:length(ann.neulay_mid)
            for i in 1:length(ann.neulay_mid[l])
                axons_mid[l][i] = sense(axons_mid[l-1], ann.neulay_mid[l][i])
            end
        end
    end

    for i in 1:length(ann.neulay_o)
        axons_o[i] = sense(axons_mid[lastindex(axons_mid)], ann.neulay_o[i])
    end

    #effect the outputing axons on recognizing
    if test == false
        return ann.effector(axons_o)
    elseif test == true
        return ANNTestdata(stimuli, axons_i, axons_mid, axons_o, ann.effector(axons_o))
    end
    #println("expect the digital is ", nn.effector(axons_o))
end

"""easy to check up everything in your ANN"""
function showANN(ann :: ANN)
    println("output layer is")
    println()
    println(ann.neulay_o)
    println()
    println("input layer is")
    println()
    println(ann.neulay_i)
    println()
    println("mid layer(s) is")
    println()
    println(ann.neulay_mid)
    println()
    println("That's all!")
end

function apply_BPargs!(ann::ANN, args::ANNBPargs)
    for i in 1:length(ann.neulay_i)
        ann.neulay_i[i].bias += args.neulay_i_b[i]
        ann.neulay_i[i].dendrites += args.neulay_i_dd[i]
    end
    for i in 1:length(ann.neulay_o)
        ann.neulay_o[i].bias += args.neulay_o_b[i]
        ann.neulay_o[i].dendrites += args.neulay_o_dd[i]
    end
    for l in 1:length(ann.neulay_mid)
        for i in 1 : length(ann.neulay_mid[l])
            ann.neulay_mid[l][i].bias += args.neulay_mid_b[l][i]
            ann.neulay_mid[l][i].dendrites += args.neulay_mid_dd[l][i]
        end
    end
    return ann
end

#endregion_general_functions
#region_hwdr

using MLDatasets.MNIST
using FileIO
using JLD2

"""hwdr means Hand-Writen Digital Recognization"""
global hwdr_ANN = creatANN((28, 28), 16, (16,), 10, effector_hwdr)

#region_hwdr_functions

"""BP means Back Propagation"""
function calBPANN_hwdr(ann::ANN, td::ANNTestdata, exp::Integer; setlong=1, plusargs=nothing, soleweight = 1, appreciate_level = 0)
    if plusargs === nothing
        args = ANNBPargs(ann)
    else
        args = plusargs
    end
    exps = zeros(Float64, length(td.axons_o))
    exps[exp+1] = 1
    #first layer
    args.neulay_o_b += 2(exps .- td.axons_o) .* (dsigmoid.(td.axons_o)) .* (setlong^-1) .* soleweight
    args.neulay_o_b = appreciate!(args.neulay_o_b, appreciate_level)
    for i in 1:length(td.axons_o)
        args.neulay_o_dd[i] += args.neulay_o_b[i] .* td.axons_mid[lastindex(td.axons_mid)] .* (setlong^-1) .* soleweight
        args.neulay_o_dd[i] = appreciate!(args.neulay_o_dd[i], appreciate_level)
    end
    #second layer(only calculate delta-bias)
    for i in 1:length(ann.neulay_mid[lastindex(ann.neulay_mid)])
        for j in 1:length(ann.neulay_o)
            args.neulay_mid_b[lastindex(ann.neulay_mid)][i] += args.neulay_o_b[j] .* ann.neulay_o[j].dendrites[i] .* dreLU.(td.axons_mid[lastindex(td.axons_mid)][i]) .* (setlong^-1) .* soleweight
            args.neulay_mid_b[lastindex(ann.neulay_mid)][i] = appreciate!(args.neulay_mid_b[lastindex(ann.neulay_mid)][i], appreciate_level)
        end
    end
    #probablely more mid layers(没来得及改完，不保证能跑)
    if length(td.axons_mid) != 1
        for l in length(td.axons_mid)-1 :-1: 1
            for i in 1:length(args.neulay_mid_b[l])
                args.neulay_mid_dd[l+1][i] += args.neulay_mid_b[l+1][i] .* td.axons_mid[l] .* (setlong^-1) .* soleweight
                args.neulay_mid_dd[l+1][i] = appreciate!(args.neulay_mid_dd[l+1][i], appreciate_level)
            end
            for i in 1:length(ann.neulay_mid[l-1])
                for j in 1:length(ann.neulay_mid[l])
                    args.neulay_mid_b[l][i] += dreLU.(td.axons_mid[l][i]) .* args.neulay_mid_b[l+1][j] .* ann.neulay_mid[l+1][j].dendrites[i] .* (setlong^-1) .* soleweight
                    args.neulay_mid_b[l][i] = appreciate!(args.neulay_mid_b[l][i], appreciate_level)
                end
            end
        end
    end
    #last layer in neulay_mid(only calculate delta-dendrites)
    for i in 1:length(td.axons_mid[1])
        args.neulay_mid_dd[1][i] += args.neulay_mid_b[1][i] .* td.axons_i .* (setlong^-1) .* soleweight
        args.neulay_mid_dd[1][i] = appreciate!(args.neulay_mid_dd[1][i], appreciate_level)
    end
    #last layer
    for i in 1:length(ann.neulay_i)
        for j in 1:length(ann.neulay_mid[1])
            args.neulay_i_b[i] += args.neulay_mid_b[1][j] .* ann.neulay_mid[1][j].dendrites[i] .* dreLU.(td.axons_i[i]) .* (setlong^-1) .* soleweight
            args.neulay_i_b[i] = appreciate!(args.neulay_i_b[i], appreciate_level)
        end
    end
    for i in 1:length(args.neulay_mid_b[1])
        args.neulay_i_dd[i] += args.neulay_i_b[i] .* td.stimuli .* (setlong^-1) .* soleweight
        args.neulay_i_dd[i] = appreciate!(args.neulay_i_dd[i], appreciate_level)
    end
    return args
end

function calcost_hwdr(data::ANNTestdata, exp::Integer)
    cost = zeros(Float64, length(data.axons_o))
    cost[exp+1] = 1
    cost = (cost .- data.axons_o) .^ 2
    return sum(cost)
end

function do1test(n)
    global hwdr_ANN
    test = Vector(undef, 2)
    test[1] = testtensor(Float64, n)
    test[2] = testlabels(n)
    testdatum = reflectANN(test[1], hwdr_ANN; test=true)
    println("axons_o is ", testdatum.axons_o)
    println("expect the digital is ", reflectANN(test[1], hwdr_ANN))
    println("actually digital is ", test[2])
    println("cost for testdatum", n, " is ", calcost_hwdr(testdatum, test[2]))
    println("\n\n\n")
end

"""do a simple check for your ANN"""
function dotestlist()
    global hwdr_ANN
    a = [2, 4, 6, 8, 3, 1, 14, 16, 18, 5]
    println("\n\n")
    for i in 1:10
        println("for ", i-1, " it gets ", reflectANN(testtensor(a[i]), hwdr_ANN))
        println("and the cost is ", calcost_hwdr(reflectANN(testtensor(a[i]), hwdr_ANN; test = true), i-1))
    end
    println("\n\n")
end

"""calculate the accuracy your hwdr_ANN does(1000)"""
function cal_accuracy(n=1000)
    global hwdr_ANN
    a = 0
    for i in 1:n
        test = Vector(undef, 2)
        test[1] = testtensor(Float64, i)
        test[2] = testlabels(i)
        testdatum = reflectANN(test[1], hwdr_ANN; test=true)
        if testdatum.expvalue == test[2]
            a += 1
        end
    end
    return a/n
end



"""train your ANN by a data set"""
function train1set(start, setn; weight = 0.1)
    global hwdr_ANN
    args = ANNBPargs(hwdr_ANN)
    for i in start:setn
        test = Vector(undef, 2)
        test[1] = traintensor(Float64, i)
        test[2] = trainlabels(i)
        testdatum = reflectANN(test[1], hwdr_ANN; test=true)
        args = calBPANN_hwdr(hwdr_ANN, testdatum, test[2]; setlong=1, soleweight = weight, appreciate_level = 0, plusargs = nothing)
    end
    global hwdr_ANN = apply_BPargs!(hwdr_ANN, args)
    println(cal_accuracy())
    return nothing
end

#endregion_hwdr_functions





dotestlist()

for j in 1 : 10
    for i in 1:20
        train1set(100i+981,100i+1321)
        println(cal_accuracy())
    end
end

println(cal_accuracy())

dotestlist()

#endregion_hwdr