
function [output_shape] = cnnOutputShape(X,conv_layer_name,n_f,n_c,s_x,s_y,s)

num_layers = length(conv_layer_name);
for i = 1:num_layers
    if conv_layer_name(i) == 1
        if i == 1
            output_hieght(i) = floor((size(X,1) - n_f(i))/(s_y{i})) + 1;
            output_width(i) = floor((size(X,2) - n_f(i) )/(s_x{i})) + 1;
            output_depth(i) = n_c(i);
        else
            output_hieght(i) = floor((output_hieght(i-1) - n_f(i) )/(s_y{i})) + 1;
            output_width(i) = floor((output_width(i-1) - n_f(i) )/(s_x{i})) + 1;
            output_depth(i) = n_c(i);       
        end
    else
        if i == 1
            output_hieght(i) = floor(size(X,1)/(s{i}));
            output_width(i) =  floor(size(X,2)/(s{i}));
            output_depth(i) = size(X,3);
        else
            output_hieght(i) = floor(output_hieght(i-1)/(s{i}));
            output_width(i) = floor(output_width(i-1)/(s{i}));
            output_depth(i) = n_c(i-1);       
        end
        
    end
end

output_shape = [output_hieght;output_width;output_depth];