#
#Author: Xlin0mu
#Date: 2022-10-22 23:55
#LastEditTime: 2022-10-23 02:08
#Description: Description
#Edited by TAMRIKer
#Copyright (c) 2022 by Xlin0mu, All Rights Reserved. 
#

#read file and preprocess it


#region_Myscope
module Myscope

using LinearAlgebra

function matmul(a1 :: Matrix{T}, a2 :: Vector{T}) where T
    return a1 .* a2
end

function matmul(a1 :: Array{T,3}, a2 :: Matrix{N}) where {T,N}
    for i in 1:lastindex(a1,3)
        a1[:, :, i] = a1[:, :, i] .* a2
    end
end

function matmul(a1 :: Array{T, 3}, a2 :: Array{T, 2}) where T
    j = lastindex(a1, 3)
    result = a1
    for i in 1 : j
        result[:, :, i] = a1[:, :, i] .* a2
    end
    return result
end

end
#endregion_Myscope

#region_ANNTrial
module ANNTrial
#a1 is toplayer
#a4 is inputlayer
#w mean weight, b mean bias
export recog, recogtest, initial_agreements, ANNagreements, Testdata, apply_dgredients, getagms, print_recogtest
using Main.Myscope
import Main.Myscope.matmul

struct Testdata
    digital :: Number
    a1 :: Array
    a2 :: Array
    a3 :: Array
    a4 :: Array
    z1 :: Array
    z2 :: Array
    z3 :: Array
    w1 :: Array
    w2 :: Array
    w3 :: Array
end

mutable struct ANNagreements
    w1 :: Array
    b1 :: Array
    w2 :: Array
    b2 :: Array
    w3 :: Array
    b3 :: Array
end

global a1, a2, a3, a4, z1, z2, z3
global agms



function initial_agreements()
    global a1, a2, a3, a4, z1, z2, z3, agms
    z3 = a4 = zeros(28, 28)
    z1 = a2 = z2 = a3 = zeros(16)
    a1 = zeros(10)
    agms = ANNagreements([0], [0], [0], [0], [0], [0])
    agms.b3 = agms.w3 = rand(28, 28, 16)
    agms.b2 = agms.w2 = rand(16, 16)
    agms.b1 = agms.w1 = rand(16, 10)
end

function getagms(agm)
    global agms
    agms = agm
end

"""
ReLU function
if num > 0 output num
else return 0
also can beused for array.
"""
function ReLU(num :: Number)
    max(0, num)
end

function dReLU(num :: Number)
    if max(0, num) > 0
        return 1
    else
        return 0
    end
end

function sigmoid(num :: Number)
    return 1 / (exp(-num) + 1)
end

function dsigmoid(num :: Number)
    snum = sigmoid(num)
    return snum * (1 - snum)
end

function calz(array :: Array{T, nd}, f :: Function) where {T, nd}
    j = lastindex(array, nd)
    result = zeros(j)
    if nd == 3
        for i in 1:j
            result[i] = f(sum(array[:, :, i]))
        end
    elseif nd == 2
        for i in 1:j
            result[i] = f(sum(array[:, i]))
        end
    else
        throw(DimensionMismatch("no methods defined for choosen dimension"))
    end
    return result
end

function recog(ra :: AbstractArray)
    global a1, a2, a3, a4, agms
    a4 = ra
    z3 = matmul(agms.w3, a4) + agms.b3
    a3 = calz(z3, ReLU)
    z2 = matmul(agms.w2, a3) + agms.b2
    a2 = calz(z2, ReLU)
    z1 = matmul(agms.w1, a2) + agms.b1
    a1 = calz(z1, sigmoid)
    digital = findmax(a1)[2] - 1
    return digital
end

function recogtest(ra :: AbstractArray)
    global a1, a2, a3, a4, agms
    a4 = ra
    z3 = matmul(agms.w3, a4) + agms.b3
    a3 = calz(z3, ReLU)
    z2 = matmul(agms.w2, a3) + agms.b2
    a2 = calz(z2, ReLU)
    z1 = matmul(agms.w1, a2) + agms.b1
    a1 = calz(z1, sigmoid)
    digital = findmax(a1)[2] - 1
    return Testdata(digital, a1, a2, a3, a4, z1, z2, z3, agms.w1, agms.w2, agms.w3)
end

function print_recogtest(td :: Testdata, nds...)
    println("digital is\n", td.digital)
    for i in nds
        println("$i is\n", td.i)
    end
    """
    println("a1 is\n", td.a1)
    println("a2 is\n", td.a2)
    println("a3 is\n", td.a3)
    println("a4 is\n", td.a4)
    elseif z == true
    println("z1 is\n", td.z1)
    println("z2 is\n", td.z2)
    println("z3 is\n", td.z3)
    elseif w == true
    println("w1 is\n", td.w1)
    println("w2 is\n", td.w2)
    println("w3 is\n", td.w3)
    """
end


end
#endregion_ANNTrial

#region_ANN_BP
module ANN_BP

export getBP123, modify_agms
using Main.ANNTrial
import Main.ANNTrial.dReLU
import Main.ANNTrial.dsigmoid



function calcost(a1, expectation)
    exp = zero(length(a1))
    exp[expectation + 1] = 1
    cost = (a1 .- exp) .^ 2
    return sum(cost)
end

function cal_exp_array(exp, length)
    exp_array = zeros(length)
    exp_array[exp + 1] = 1
    return exp_array
end

"""
at least it's could be used for an ANN like
input layer as 28 * 28
2 hiden layer, both are 16-length Vector
output layer as 10-length Vector
Won't be sured for applying to other ANN
"""
function getBP123(td :: Testdata)
    expa = cal_exp_array(td.digital, length(td.a1))
    gweight1 = gbias1 = td.w1
    gweight2 = gbias2 = td.w2
    gweight3 = gbias3 = td.w3

    gbias1 = dsigmoid.(td.z1) .* 2(expa .- td.a1)'
    gweight1 = gbias1 .* td.a2
    for i in 1:size(gbias1)[1]
        td.a2[i] = sum((gbias1 .* td.w1)[i,:])
    end

    gbias2 = dReLU.(td.z2) .* td.a2'
    gweight2 = gbias2 .* td.a3
    for i in 1:size(gbias2)[1]
        td.a3[i] = sum((gbias2 .* td.w2)[i,:])
    end

    for i in size(td.z3)[3]
        gbias3[:,:,i] = dReLU.(td.z3)[:, :, i] .* td.a3'[i]
    end
    for i in size(gweight3)[3]
        gweight3[:, :, i] = gbias3[:, :, i] .* td.a4
    end
    return ANNagreements(-gweight1, -gbias1, -gweight2, -gbias2, -gweight3, -gbias3)
end

function modify_agms(agms :: ANNagreements, dg :: ANNagreements)
    agms.w1 += dg.w1
    agms.w2 += dg.w2
    agms.w3 += dg.w3
    agms.b1 += dg.b1
    agms.b2 += dg.b2
    agms.b3 += dg.b3
    return agms
end



end
#endregion_ANN_BP




using Main.ANNTrial
using Main.ANN_BP
using MLDatasets.MNIST

initial_agreements()

a4 = traintensor(Float64, 7)


println("testing")

"""
for i in 1:100
    getagms(modify_agms(Main.ANNTrial.agms, getBP123(recogtest(traintensor(Float64, i)))))
end
"""



print_recogtest(recogtest(a4), :a1, :a2, :a3, :w1, :w2, :z1, :z2)

println("success!")