function simple_render_pendulum(state; show_prev = true, dt = 1, down = true, stride = 4)
    """
    State from openai gym is [cos(theta), sin(theta), theta_dot]
    """
    θ_curr = state[1] #atan(state[2], state[1])
    if show_prev
        θ_prev = θ_curr - state[2]*dt #state[3]*dt
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
    rot_mat_cw = @SArray [cos(θ) sin(θ); -sin(θ) cos(θ)]
    height = 70
    width = 10
    # center = height + width
    # shift = @SArray [center, width]

    # Empty matrix to store full size image before cropping and downsampling
    pic = ones(4*width, height)
    shift = [Int(size(pic, 1) / 2), 0]
    
    # Loop through all pendulum indices
    for point in Iterators.product(-width/2:width/2, 0:height)
        # Rotate to correct angle
        rotated_point = round.(Int, rot_mat_cw * collect(point))
        # Shift to location in pic array (can't have negative indices)
        shifted_point = rotated_point .+ shift
        # Make that index black
        if shifted_point[1] > 1 && shifted_point[1] <= size(pic, 1) &&
            shifted_point[2] > 1 && shifted_point[2] <= size(pic, 2)
            pic[shifted_point[1], shifted_point[2]] = 1 - intensity
        end
    end

    # Crop to 400x600
    # cropped_pic = pic[center - 200:center + 199, height - 599:height]

    return pic
end

