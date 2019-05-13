 

function O = Forward_MaxPool(X,s)
% X - Input - X -  n_x X n_y X n_h X m_batch
% s - stride - same stride in x and y assumed 

m = size(X,4);
for training_sample = 1:m
    O(:,:,:,training_sample) = max_pool(X(:,:,:,training_sample),s);
end 