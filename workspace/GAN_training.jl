using Flux
using Parameters
using Printf
using Random
using Statistics
using Images

using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Flux.Optimise: train!, @epochs
using Base.Iterators: partition

using BSON
using Distributions

@with_kw struct HyperParameters
	batch_size::Int = 128
	latent_dim::Int = 100
	epochs::Int = 20
	verbose_freq::Int = 1000
	output_x::Int = 10    # for visualization of generator
	output_y::Int = 10    # for visualization of generator
	αᴰ::Float64 = 0.0002 # discriminator learning rate
	αᴳ::Float64 = 0.0002 # generator learning rate
    device::Function = cpu # device to send operations to. Can be cpu or gpu
    
    # Generator architecture parameters - default 128, 128, 64
    latent_channels::Int = 128
    conv_1_channels::Int = 128
    conv_2_channels::Int = 64
end

############## Discriminator structure #################################

struct Discriminator
    # Submodel to take labels as input and convert them to the shape of image ie. (10, 18, 1, batch_size)
    d_labels
    d_common
end

function Discriminator(hp::HyperParameters)
	d_labels = Chain(Dense(2, 180), x-> reshape(x, 10, 18, 1, size(x, 2))) |> hp.device
    d_common = Chain(Conv((3,3), 2=>128, pad=(1,1), stride=(2,2)),
                  x-> leakyrelu.(x, 0.2f0),
                  Dropout(0.4),
                  Conv((3,3), 128=>128, pad=(1,1), stride=(2,2), leakyrelu),
                  x-> leakyrelu.(x, 0.2f0),
                  x-> reshape(x, :, size(x, 4)),
                  Dropout(0.4),
                  Dense(1920, 1)) |> hp.device
    return Discriminator(d_labels, d_common)
end

function (m::Discriminator)(x, y)
    # println(size(m.d_labels(y)))
    # println(size(x))
    t = cat(m.d_labels(y), x, dims=3)
    return m.d_common(t)
end

############## Generator structure #################################

struct Generator
    # Submodel to take labels as input and convert it to the shape of (3, 5, 1, batch_size)
    g_labels          
    # Submodel to take latent_dims as input and convert it to shape of (3, 5, 128, batch_size)
    g_latent          
    g_common
end

function Generator(hp::HyperParameters)
    latent_im_nodes = 3*5
    g_labels = Chain(Dense(2, latent_im_nodes), x-> reshape(x, 3, 5, 1 , size(x, 2))) |> hp.device
    g_latent = Chain(Dense(hp.latent_dim, hp.latent_channels * latent_im_nodes), 
            x-> leakyrelu.(x, 0.2f0), 
            x-> reshape(x, 3, 5, hp.latent_channels, size(x, 2))) |> hp.device
    g_common = Chain(ConvTranspose((3, 3), hp.latent_channels+1=>hp.conv_1_channels; stride=2, pad=1),
            BatchNorm(hp.latent_channels, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), hp.conv_1_channels=>hp.conv_2_channels; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), hp.conv_2_channels=>1, tanh; stride=1, pad=3)) |> hp.device
    return Generator(g_labels, g_latent, g_common)
end

function (m::Generator)(x, y)
    t = cat(m.g_labels(y), m.g_latent(x), dims=3)
    return m.g_common(t)
end

############## Training helpers #################################

function Lᴰ(real_output, fake_output)
	real_loss = logitbinarycrossentropy(real_output, 1f0, agg=mean)
	fake_loss = logitbinarycrossentropy(fake_output, 0f0, agg=mean)
	return real_loss + fake_loss
end

Lᴳ(fake_output) = logitbinarycrossentropy(fake_output, 1f0, agg=mean)

function train_discriminator!(G, D, nx, ny, x, y, optD)
    θ = Flux.params(D.d_labels, D.d_common)
    loss, back = Flux.pullback(() -> Lᴰ(D(x, y), D(G(nx, ny), ny)), θ)
    update!(optD, θ, back(1f0))
    return loss
end

function train_generator!(G, D, nx, ny, optG)
	θ = Flux.params(G.g_labels, G.g_latent, G.g_common)
	loss, back = Flux.pullback(() -> Lᴳ(D(G(nx, ny), ny)), θ)
	update!(optG, θ, back(1f0))
	return loss
end

############## Util #################################
function to_image(G, fixed_noise, fixed_labels, hp)
    fake_images = cpu.(G.(fixed_noise, fixed_labels))
    image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hp.output_y))); dims=(3, 4)), (2, 1))
    image_array = Gray.(image_array .+ 1f0) ./ 2f0
    return image_array
end

# Function that returns random input for the generator
function rand_input(hp)
   x = randn(Float32, hp.latent_dim, hp.batch_size) |> hp.device
   y = Float32.(vcat(rand(Uniform(-22, 22), 1, hp.batch_size), rand(Uniform(-8, 8), 1, hp.batch_size))) |> hp.device
   return x, y
end

############## Training #################################
function train(fn ;hp = HyperParameters(), G = Generator(hp), D = Discriminator(hp), optG = ADAM(hp.αᴳ, (0.5, 0.99)), optD = ADAM(hp.αᴰ, (0.5, 0.99)))
    # Load MNIST dataset
    res = BSON.load(fn)
    images = res[:images]
    images = reshape(2f0 .* images .- 1f0, 10, 18, 1, :) |> hp.device # Normalize to [-1, 1]
    y = Float32.(res[:y]) |> hp.device
    data = DataLoader((images, y), batchsize=hp.batch_size, shuffle = true)

    fixed_noise = [randn(hp.latent_dim, 1) |> hp.device for _=1:hp.output_x * hp.output_y]
    fixed_labels = [Float32.([rand(Uniform(-22, 22)), rand(Uniform(-8, 8))]) |> hp.device
                             for _ =1:hp.output_x * hp.output_y]

    # Training
    step = 0
	@epochs hp.epochs for (x, y) in data
        loss_D = train_discriminator!(G, D, rand_input(hp)..., x, y, optD)
		loss_G = train_generator!(G, D, rand_input(hp)..., optG)
        if step % hp.verbose_freq == 0
            @info("Train step $(step), Discriminator loss = $loss_D, Generator loss = $loss_G)")
            #save(@sprintf("output/cgan_steps_%06d.png", step), to_image(G, fixed_noise, fixed_labels, hp))
        end
        step += 1
    end
	return G
end

fn = "/scratch/smkatz/NASA_ULI/pendulum_GAN_data.bson"
G_conv = train(fn)