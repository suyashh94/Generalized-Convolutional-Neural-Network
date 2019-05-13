
function dFilt = convFilterGrad(dZ,I,filt,s_x,s_y)
%dZ - Gradient of output of one layer - o_h X o_w X n_c X m_batch  - dimension of output
%I - Input image for this layer of convolution - i_h X i_w X i_n X m_batch
% filt - filter used to get this layer of output - n_f X n_f X i_n X n_c
% s_x , s_y = stride for this layer of output

m = size(dZ,4);
o_h = size(dZ,1);
o_w = size(dZ,2);
n_h = size(I,3);
n_f = size(filt,1);
n_channels = size(filt,4);
dFilt = zeros(size(filt));

for training_sample = 1 : m
    for channels = 1:n_channels
        for out_height = 1:o_h
            for out_width = 1:o_w
                input = dZ(out_height,out_width,channels,training_sample);
                x_range = (out_width - 1)*s_x + 1 : (out_width - 1)*s_x + 1 + n_f - 1;
                y_range = (out_height - 1)*s_y + 1 : (out_height - 1)*s_y + 1 + n_f - 1;
                dFilt(:,:,:,channels) = dFilt(:,:,:,channels) + I(y_range,x_range,:,training_sample) .* input;
            end
        end
    end
end
