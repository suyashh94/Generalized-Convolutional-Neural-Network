
function [filt,bias] = update_parameters_conv(filt,bias,learning_rate,dFilt,dBias,conv_layer_name)

num_layers = length(filt);

for i = 1:num_layers
    if conv_layer_name(i) == 1
        filt{i} = filt{i} - learning_rate * dFilt{i};
        bias{i} = bias{i} - learning_rate * dBias{i};
    end
end