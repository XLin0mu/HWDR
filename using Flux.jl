#
#Author: Xlin0mu
#Date: 2022-10-22 23:55
#LastEditTime: 2022-10-23 02:08
#Description: Description
#Edited by TAMRIKer
#Copyright (c) 2022 by Xlin0mu, All Rights Reserved. 
#

using Flux

function flux_to_myANN(ann_f :: Chain, effector, disposions...)
    l = length(ann_f)
    if length(disposions) != l-1
        throw(ArgumentError("disposions' length get $(length(disposions)), but function require $(l-1))"))
    end
    layers = Vector{ANNLayer}(undef, l-1)
    for i in 1 : l-1
        layers[i] = ANNLayer((size(ann_f[i].weight')), ann_f[i].bias, ann_f[i].weight, disposions[i])
    end
    ANN(layers, effector)   
end
