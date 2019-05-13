function dI = convInputGrad(dZ,I,filt,s_x,s_y)

%dZ - Gradient of output of one layer - o_h X o_w X n_c X m_bacth  - dimension of output
%I - Input image for this layer of convolution - i_h X i_w X i_n X m_batch
% filt - filter used to get this layer of output - n_f X n_f X i_n X n_c
% s_x , s_y = stride for this layer of output

m = size(dZ,4);
n_channels = size(filt,4);

i_n = size(I,3);
i_h = size(I,1);
i_w = size(I,2);

o_h = size(dZ,1);
o_w = size(dZ,2);

n_f = size(filt,1);

assert(size(filt,3) == i_n);
dI = zeros(size(I));
for training_sample = 1:m
    for channels = 1:n_channels
        for i = 1:o_h
            for j = 1:o_w
                input = dZ(i,j,channels,training_sample);
                prod = input * filt(:,:,:,channels);
                x_range = (j-1)*s_x + 1 : (j-1)*s_x + 1 + n_f - 1;
                y_range = (i-1)*s_y + 1 : (i-1)*s_y + 1 + n_f - 1;
                dI(y_range,x_range,:,training_sample) = dI(y_range,x_range,:,training_sample) + prod;
            end
        end
    end
end