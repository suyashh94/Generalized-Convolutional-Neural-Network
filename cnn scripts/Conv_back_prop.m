
function [dA, dZ, dFilt, dBias] = Conv_back_prop(dZ,Z,A,X,filt,s_x,s_y,s,conv_layer_name,max_mask)
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
        dBias{l} = convBiasGrad(dZ{l});
        if l > 1
            dA{l-1} = convInputGrad(dZ{l},A{l-1},filt{l},s_x{l},s_y{l});
            dZ{l-1} = dA{l-1} .* linear_activation_grad(Z{l-1});
        end
    elseif conv_layer_name(l) == 2
        dFilt{l} = 0;
        dBias{l} = 0;
        if l > 1
            dA{l-1} = maxPoolGrad(dZ{l},A{l-1},max_mask{l-1},s{l});
            dZ{l-1} = dA{l-1};
        end
    end
end

