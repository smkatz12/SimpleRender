function simple_render_bgw(state; bgw_size = 10, pit_min = 5, pit_max = 6, background_range = 0.1)
    """
    NOTE: this function is stochastic!
    State is:
     - x-position (between 1 and bgw_size) :: Int
     - y-position (between 1 and bgw_size) :: Int
    Pit will be located between pit_min and pit_max in x and y directions
    Noise sampled in background will be applied to all other pixels in the image
    """

    x, y = state
    dist = Distributions.Uniform(1.0 - background_range, 1.0)

    pixels = ones(bgw_size, bgw_size)
    for i = 1:bgw_size
        for j = 1:bgw_size
            if i == y && j == x
                pixels[i, j] = 0.0
            elseif pit_min ≤ i ≤ pit_max && pit_min ≤ j ≤ pit_max
                pixels[i, j] = 0.0
            else
                pixels[i, j] = rand(dist)
            end
        end
    end
    return pixels
end

function simple_render_gw(state; bgw_size = 10, pit_min = 5, pit_max = 6)
    """
    Renders grid world image with no noise
    State is:
     - x-position (between 1 and bgw_size) :: Int
     - y-position (between 1 and bgw_size) :: Int
    Pit will be located between pit_min and pit_max in x and y directions
    Noise sampled in background will be applied to all other pixels in the image
    """

    x, y = state

    pixels = ones(bgw_size, bgw_size)
    for i = 1:bgw_size
        for j = 1:bgw_size
            if i == y && j == x
                pixels[i, j] = 0.0
            elseif pit_min ≤ i ≤ pit_max && pit_min ≤ j ≤ pit_max
                pixels[i, j] = 0.0
            end
        end
    end
    return pixels
end