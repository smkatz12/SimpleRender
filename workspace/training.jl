using POMDPs, Shard, Flux, POMDPPolicies, Plots, BSON

include("pendulum_mdp.jl")
include("../simple_render.jl")
using POMDPGym

s = [0.5, 0.5]
@time img = simple_render_pendulum(s, stride = 4)
heatmap(img')

as = [-2., -0.5, 0, 0.5, 2.]
mdp = InvertedPendulum(actions = as, pixel_observations = true, render_fun = simple_render_pendulum)
s_dim = length(rand(observation(mdp, rand(initialstate(mdp)))))

scale2(x) = (x .- [0f0, 0f0]) ./ [Float32(mdp.failure_thresh), 8f0]
scale3(x) = x ./ Float32[1, 1, 8]
Q() = Chain(Dense(s_dim, 64, relu), Dense(64,32,relu), Dense(32, length(as)), (x) -> 50f0 * x .+ 50f0)
N = 100000
pol = DQNPolicy(Q(), as)
solver = DQNSolver(π = pol, sdim = s_dim, N = N, opt = ADAM(1e-5), eval_eps = 5)
solve(solver, mdp)

BSON.@save "image_policy.bson" pol


sampler = Sampler(mdp, pol, s_dim, length(as), max_steps = 100)
data, eps = episodes!(sampler, return_episodes = true)
undiscounted_return(data, first(eps)...)
# scatter(acos.(data[:s][1, :]), data[:s][3, :])
scatter(data[:s][1, :], data[:s][2, :])



