function dI = maxPoolGrad(dZ,I,masks,s)

%dZ - Gradient of output of one layer - o_h X o_w X n_c X m_bacth  - dimension of output
%I - Input image for this layer of convolution - i_h X i_w X i_n X m_batch
% masks - i_h X i_w X i_n X m_batch
% s = stride of max for this layer

m = size(dZ,4);
n_channels = size(dZ,3);

i_n = size(I,3);
i_h = size(I,1);
i_w = size(I,2);

o_h = size(dZ,1);
o_w = size(dZ,2);

dI = zeros(size(I));
for training_sample = 1:m
    for channels = 1:n_channels
        for i = 1:o_h
            for j = 1:o_w
                input = dZ(i,j,channels,training_sample);
                x_range = (j-1)*s + 1 : (j-1)*s + s;
                y_range = (i-1)*s + 1 : (i-1)*s + s;
                dI(y_range,x_range,channels,training_sample) = input;
            end
        end
    end
end

dI = dI .* masks;