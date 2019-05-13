
function O = Forward_Convolution(X,filt,s_x,s_y)

% X -  n_x X n_y X n_h X m_batch
% filt =  n_f X n_f X n_h X n_c

m = size(X,4);
n_channels = size(filt,4);


for training_sample = 1:m
    for channels = 1:n_channels
        O(:,:,channels,training_sample) = filterConv(X(:,:,:,training_sample),filt(:,:,:,channels)...
            ,s_x,s_y);
    end
end