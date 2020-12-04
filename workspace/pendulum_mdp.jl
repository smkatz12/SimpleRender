using POMDPGym, Parameters, Random, Distributions, POMDPModelTools

const S = Array{Float64}
const O = Array{Float32}
const A = Float64

@with_kw struct InvertedPendulum <: POMDP{S, A, O}
    failure_thresh::Float64 = deg2rad(20)
    max_speed::Float64 = 8.
    max_torque::Float64 = 2.
    dt::Float64 = .05
    g::Float64 = 10.
    m::Float64 = 1.
    l::Float64 = 1.
    γ::Float64 = 0.99
    actions::Vector{A} = [-1., 1.]
    pixel_observations::Bool = false
    render_fun::Union{Nothing, Function} = nothing
end

angle_normalize(x) = mod((x+π), (2*π)) - π

function POMDPs.gen(mdp::InvertedPendulum, s::S, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    θ, ω = s[1], s[2]
    dt, g, m, l = mdp.dt, mdp.g, mdp.m, mdp.l

    a = clamp(a, -mdp.max_torque, mdp.max_torque)
    costs = angle_normalize(θ)^2 + 0.1 * ω^2 + 0.001 * a^2

    ω = ω + (-3. * g / (2 * l) * sin(θ + π) + 3. * a / (m * l^2)) * dt
    θ = θ + ω * dt
    ω = clamp(ω, -mdp.max_speed, mdp.max_speed)

    sp = [θ, ω]
    (sp = sp, o=rand(rng, observation(mdp, sp)), r = 1 - 0.01*costs)
end

function POMDPs.observation(mdp::InvertedPendulum, s::S)
    o = mdp.pixel_observations ? mdp.render_fun(s)[:] : [angle_normalize(s[1]), s[2]] #[cos(s[1]), sin(s[1]), s[2]]
    Deterministic(Float32.(o))
end

function POMDPs.initialstate(mdp::InvertedPendulum, rng::AbstractRNG = Random.GLOBAL_RNG)
     θ = rand(rng, Distributions.Uniform(- mdp.failure_thresh/2., mdp.failure_thresh/2.))
     ω = rand(rng, Distributions.Uniform(-.1, .1))
     # θ = rand(rng, Distributions.Uniform(-π, π))
     # ω = rand(rng, Distributions.Uniform(-1., 1.))
     Deterministic([θ, ω])
 end
POMDPs.initialobs(mdp::InvertedPendulum, s::S)  = observation(mdp, s)

POMDPs.actions(mdp::InvertedPendulum) = mdp.actions

POMDPs.isterminal(mdp::InvertedPendulum, s::S) = abs(s[1]) > mdp.failure_thresh
POMDPs.discount(mdp::InvertedPendulum) = mdp.γ

