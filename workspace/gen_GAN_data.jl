using Distributions
using BSON
using BSON: @save

include("../simple_render.jl")

num_points = 10000

θdist = Uniform(-22, 22)
ωdist = Uniform(-8, 8)

images = zeros(10, 18, num_points)
y = zeros(2, num_points)

for i = 1:num_points
    # Draw sample
    y[1, i] = rand(θdist)
    y[2, i] = rand(ωdist)
    
    images[:, :, i] = simple_render_pendulum(y[:, i])
end

@save "/scratch/smkatz/NASA_ULI/pendulum_GAN_data.bson" images y