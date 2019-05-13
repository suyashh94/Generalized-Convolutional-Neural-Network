
function [O ,A] = Forward_Convolution_And_Relu(X,filt,bias,s_x,s_y)

% X -  n_x X n_y X n_h X m_batch
% filt =  n_f X n_f X n_h X n_c
% bias = n_c X 1

m = size(X,4);
n_channels = size(filt,4);


for training_sample = 1:m
    
    for channels = 1:n_channels
%         channels
        O(:,:,channels,training_sample) = filterConv(X(:,:,:,training_sample),filt(:,:,:,channels)...
            ,s_x,s_y);
        O(:,:,channels,training_sample) = O(:,:,channels,training_sample) + bias(channels);
    end
end

A = linear_activation(O);