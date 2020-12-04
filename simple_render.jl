using Statistics
using Distributions
using StaticArrays

include("pendulum.jl")
include("blurry_grid_world.jl")

function downsample(pic; stride = 50, avg = true)
    height = convert(Int64, ceil(size(pic, 1) / stride))
    width = convert(Int64, ceil(size(pic, 2) / stride))

    downsampled_pic = zeros(height, width)

    for i = 1:width
        colmin = stride * (i - 1) + 1
        colmax = min(stride * i, size(pic, 2))
        for j = 1:height
            rowmin = stride * (j - 1) + 1
            rowmax = min(stride * j, size(pic, 1))
            if avg
                downsampled_pic[j, i] = mean(pic[rowmin:rowmax, colmin:colmax])
            else
                downsampled_pic[j, i] = minimum(pic[rowmin:rowmax, colmin:colmax])
            end
        end
    end

    return downsampled_pic
end
