
function toKeep_matrix = getDropout_keepMatrix(conv_layer_name,output_shape,minibatch_size,keep_prob)

% output shape - size of each layer in rows X number of layers

num_layers = size(output_shape,2);
m = minibatch_size;

for layer = 1:num_layers
    toKeep_matrix{layer} = ones(output_shape(1,layer),output_shape(2,layer),...
        output_shape(3,layer),m);
end

for layer = 1:num_layers
    prob_matrix{layer} = rand(output_shape(3,layer),m);
    prob_matrix{layer} = (prob_matrix{layer} <= keep_prob);
end

for layer = 1:num_layers
    [channel{layer},sample{layer}] = find(prob_matrix{layer} == 0);
end


for layer = 1:num_layers
    if conv_layer_name(layer) == 1
        for j = 1:length(channel{layer})
            toKeep_matrix{layer}(:,:,channel{layer}(j),sample{layer}(j)) = 0;
        end
    end
end

