using Distributions
using BSON
using BSON: @save

include("../simple_render.jl")

num_points = 10000

θdist = Uniform(-deg2rad(22), deg2rad(22))
ωdist = Uniform(-8, 8)

images = zeros(Float32, 10, 18, num_points)
y = zeros(Float32, 2, num_points)

for i = 1:num_points
    # Draw sample
    y[1, i] = rand(θdist)/deg2rad(22)
    y[2, i] = rand(ωdist)/8
    
    images[:, :, i] = simple_render_pendulum(y[:, i])
end

@save "/scratch/smkatz/NASA_ULI/pendulum_GAN_data_norm.bson" images y