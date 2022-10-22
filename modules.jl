









#region_Universal

module Universal



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

function reLU(arg::Real; d = 0)
    if d == 0||"dx"||"dy"
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
    return nothing
end

function sigmoid(arg::Real; d = 0)
    if d == 0||"dx"||"dy"
        if d == "dy"
            return d*(1-d)
        end
        else snum = 1 / (1 + ℯ^(-arg))
            if d == 0
                return snum
            else
                return snum*(1-snum)
    end
    return nothing
end

function donothing(a)
    return a
end



end

#endregion_Universal


