
function [A, Z, max_mask] = Conv_forward_prop_dropout(X,conv_layer_name,filt,bias,s_x,s_y,s,toKeep_matrix,keep_prob)

% X - Input image of shape - i_h X i_w X i_n X m_batch
% conv_layer_name - double defining a layer to be a conv+relu or max pool
% filt - filter for all layers
% bias - bias for all filts in each layer
% s_x - x stride for all layers
% s_y - y stride for all layers
% s - max stride for all max pool layers


A_prev = X;

num_conv_layers = length(conv_layer_name);

for l = 1:num_conv_layers
    if conv_layer_name(l) == 1
        [Z{l} ,A{l}] = Forward_Convolution_And_Relu_dropout(A_prev,filt{l},bias{l},s_x{l},s_y{l},toKeep_matrix{l},keep_prob);
        if l < length(conv_layer_name)
            max_mask{l} = max_pool_masks(A{l},s{l+1});
        else
            max_mask{l} = max_pool_masks(A{l},1);
        end
        
    elseif conv_layer_name(l) == 2
        Z{l} = Forward_MaxPool(A_prev,s{l});
        A{l} = Z{l};
        if l < length(conv_layer_name)
            max_mask{l} = max_pool_masks(A{l},s{l+1});
        else
            max_mask{l} = max_pool_masks(A{l},1);
        end
    end
    A_prev = A{l};
end