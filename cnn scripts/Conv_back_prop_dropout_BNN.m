
function [dA, dZ, dFilt, d_gamma, d_beta] = Conv_back_prop_dropout_BNN...
    (dZ,A,X,filt,s_x,s_y,s,conv_layer_name,max_mask,toKeep_matrix,keep_prob...
    ,Z_bn_unfolded,Z_hat_unfolded,Z_mu_unfolded,gamma,...
    ivar, var_sqrt, var, eps)
% dZ - gradient of last conv layer - i_h X i_w X i_n X m_batch
% filt - filters at all layers
% bias - bias at all layers
% s_x - x stride all layers
% s_y - y stride all laters
% s - max pool stride all layers
% conv_layer_name - name of all conv layers
% max_mask - max mask at all layers

num_conv_layers = length(conv_layer_name);

for l = num_conv_layers : -1:1
    if conv_layer_name(l) == 1
        if l > 1
            dFilt{l} = convFilterGrad(dZ{l},A{l-1},filt{l},s_x{l},s_y{l});
        else
            dFilt{l} = convFilterGrad(dZ{l},X,filt{l},s_x{l},s_y{l});
        end
        if l > 1
            dA{l-1} = convInputGrad(dZ{l},A{l-1},filt{l},s_x{l},s_y{l}).* toKeep_matrix{l-1};
            dA{l-1} = dA{l-1} ./ keep_prob;
           [dZ{l-1},d_gamma{l-1},d_beta{l-1}] = ...
               batch_norm_backward_cnn(dA{l-1},Z_bn_unfolded{l-1},Z_hat_unfolded{l-1},...
               gamma{l-1}, Z_mu_unfolded{l-1}, ivar{l-1}, var_sqrt{l-1}, var{l-1}, eps{l-1});
        end
    elseif conv_layer_name(l) == 2
        dFilt{l} = 0;
        if l > 1
            dA{l-1} = maxPoolGrad(dZ{l},A{l-1},max_mask{l-1},s{l});
            [dZ{l-1},d_gamma{l-1},d_beta{l-1}] = ...
               batch_norm_backward_maxpool(dA{l-1},Z_hat_unfolded{l-1},...
               gamma{l-1}, Z_mu_unfolded{l-1}, ivar{l-1}, var_sqrt{l-1}, var{l-1}, eps{l-1});
        end
    end
end

