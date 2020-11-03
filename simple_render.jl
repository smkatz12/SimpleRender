using Statistics

function simple_render_pendulum(state; show_prev = true, dt = 1, down = true, stride = 36)
    """
    State from openai gym is [cos(theta), sin(theta), theta_dot]
    """
    θ_curr = atan(state[2], state[1])
    if show_prev
        θ_prev = θ_curr - state[3]*dt
    end

    curr_frame = full_resolution_pendulum(θ_curr)
    if show_prev
        prev_frame = full_resolution_pendulum(θ_prev, intensity = 0.5)
        curr_frame = min.(prev_frame, curr_frame)
    end

    if down
        curr_frame = downsample(curr_frame, stride = stride)
    end

    return curr_frame
end

function full_resolution_pendulum(θ; intensity = 1)
    rot_mat_cw = [cos(θ) sin(θ); -sin(θ) cos(θ)]
    height = 1300
    width = 40
    center = height + width

    # Empty matrix to store full size image before cropping and downsampling
    pic = ones(2 * (height + width), height + width)

    # Loop through all pendulum indices
    for point in Iterators.product(-width/2:width/2, 0:height)
        # Rotate to correct angle
        rotated_point = round.(Int, rot_mat_cw * collect(point))
        # Shift to location in pic array (can't have negative indices)
        shifted_point = rotated_point .+ [center, width]
        # Make that index black
        pic[shifted_point[1], shifted_point[2]] = 1 - intensity
    end

    # Crop to 400x600
    cropped_pic = pic[center - 200:center + 199, height - 599:height]

    return cropped_pic
end

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
