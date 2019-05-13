
function [Z,A,Z_bn_unfolded,Z_hat_unfolded,gamma,...
            Z_mu_unfolded, ivar, var_sqrt, var, eps,max_mask] = Conv_forward_prop_BN_Test...
            (X,conv_layer_name,filt,s_x,s_y,s,gamma,beta,mu,sigma)

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
        [Z{l},A{l},Z_bn_unfolded{l},Z_hat_unfolded{l},gamma{l},...
            Z_mu_unfolded{l}, ivar{l}, var_sqrt{l}, var{l}, eps{l}] = ...
    Forward_Convolution_And_Relu_BNN_Test(A_prev,filt{l},s_x{l},s_y{l},gamma{l},beta{l},mu{l},sigma{l});
        if l < length(conv_layer_name)
            max_mask{l} = max_pool_masks(A{l},s{l+1});
        else
            max_mask{l} = max_pool_masks(A{l},1);
        end
        
    elseif conv_layer_name(l) == 2
        Z{l} = Forward_MaxPool(A_prev,s{l});
        [A{l},Z_bn_unfolded{l},Z_hat_unfolded{l},gamma{l},Z_mu_unfolded{l}, ivar{l}, var_sqrt{l}, var{l}, eps{l}] ...
            = batch_norm_forward_maxpool_Test(Z{l},gamma{l},beta{l},mu{l},sigma{l});
        if l < length(conv_layer_name)
            max_mask{l} = max_pool_masks(A{l},s{l+1});
        else
            max_mask{l} = max_pool_masks(A{l},1);
        end
    end
    A_prev = A{l};
end