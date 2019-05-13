clear all;
%%

% load('C:\Users\harla\Desktop\Matlab scripts\Datasets\Mnist\training_data.mat')
load('C:\Users\harla\Desktop\Matlab scripts\Datasets\MNIST Fashion\training_set.mat')


%% X train and Y train
x_train = double(x_train);
x_train = reshape(x_train,[28,28,1,60000]);
% x_train = permute(x_train,[2 1 3 4]);    % This statement for MNIST
y_train_hot = zeros(10,size(x_train,4)); 
for i = 1:size(x_train,4)
    y_train_hot(y_train(i)+1,i) = 1; 
end
y = y_train_hot;

% y = y_train                               % This statement for MNIST


t = 1:60000;
t = shuffle(t);

X_train_main = x_train(:,:,:,t);
y_train_main = y(:,t);

minibatch_size = 10;
for k = 1:size(y_train_main,2)/minibatch_size
    range = (k-1)*minibatch_size + 1 : k*minibatch_size;
    X_minibatch{k} = X_train_main(:,:,:,range);
    y_train_minibatch{k} = y_train_main(:,range);
end
%% Model Architecture

num_conv_layer = 4;

for i = 1:num_conv_layer
    conv_layer_name(i) = input('Layer type - \n 1 for conv + relu \n 2 for max pool \n :  ');
end

%% Defining strides for every layer

for i = 1:num_conv_layer
    if conv_layer_name(i) == 1
        s_x{i} = 1;
        s_y{i} = 1;
        s{i} = 1;
    else
        s_x{i} = 0;
        s_y{i} = 0;
        s{i} = 2;
    end
end

%% Number of filters to have in each layer
for i = 1 :num_conv_layer
    if conv_layer_name(i) == 1
        n_c(i) = input(['No. of filters in layer number ',num2str(i), ' - conv layer']);
    else
        n_c(i) = 0;
    end
end

%% Filter size for every convolution layer
for i = 1:num_conv_layer
    if conv_layer_name(i) == 1
        n_f(i) = input(['Filter size',num2str(i), ' - conv layer']);
    else
        n_f(i) = 0;
    end
end
%% Initializing weights of filters

convLayer = find(conv_layer_name == 1);
count = 0;
for i = 1:num_conv_layer
    if conv_layer_name(i) == 1
        if i == 1
            filt{i} = rand(n_f(i),n_f(i),size(X_train_main,3),n_c(i)) * 1;
%             filt{i} = filt{i} * sqrt(2/numel(filt{i}));
            count = count + 1;
        else
            filt{i} = rand(n_f(i),n_f(i),n_c(convLayer(count)),n_c(i)) * 0.01;
            filt{i} = filt{i} * sqrt(2/numel(filt{i}));
            count = count + 1;
        end
    else
        filt{i} = 0;
    end
end

%% Initializing bias for filter

convLayer = find(conv_layer_name == 1);
count = 0;
for i = 1:num_conv_layer
    if conv_layer_name(i) == 1
        bias{i} = zeros(n_c(i),1);
    else
        bias{i} = 0;
    end
end

%% Initialising momentum terms for filters
for i = 1:num_conv_layer
    if conv_layer_name(i) == 1
        V_dFilt{i} = zeros(size(filt{i}));
    else
        V_dFilt{i} = 0;
    end
end
%% Initialising momentum terms for Bias
for i = 1:num_conv_layer
    if conv_layer_name(i) == 1
        V_dBias{i} = zeros(size(bias{i}));
    else
        V_dBias{i} = 0;
    end
end

%% Out put shape of CNN layers
[output_shape] = cnnOutputShape(x_train,conv_layer_name,n_f,n_c,s_x,s_y,s);
%% FC layer architecture
num_layers = 1; % Number of layers excluding the input / unfoldfing layer , including the output layer

layer_unfolded_numel = output_shape(1,end)*output_shape(2,end)*output_shape(3,end);
layer_dims = zeros(num_layers+1,1);
layer_dims(1) = layer_unfolded_numel; % This needs to be generalized. In this case it is 150
layer_dims(end) = size(y,1); % This needs to be generalized too. In this case this is 10
for i = 2:length(layer_dims)-1
    layer_dims(i) = input(['Enter number of units in layer ' num2str(i)]);
end

%% Initializing FC weights and bias
[W_fc, b_fc] = initialize_parameters(layer_dims);
%% Initialize momentum terms for FC weights and bias
[V_dW_fc, V_db_fc] = initialize_parameters_with_zero(layer_dims);
%% Defining activations for FC
% Last layer has activation sigmoid
% All layers before it has activation relu
for i = 1:num_layers
    if i ~= num_layers
        if mod(i,2) == 0
            activation{i} = 'relu';
        else
            activation{i} = 'sigmoid';
        end
    else
        activation{i} = 'sigmoid';
    end
end
%% Constants

learning_rate = 0.02;
%% Iteration starts
for iter = 1:100
    for k = 1:10%length(X_minibatch)
        k
        disp(['iteration number ', num2str(iter)])
        X = X_minibatch{k};
        y_train = y_train_minibatch{k};
        %% Forward propragation for convolution layers
        [A, Z, max_mask] = Conv_forward_prop(X,conv_layer_name,filt,bias,s_x,s_y,s);
        
        %% Unfold the final conv layer out and forward prop using a fully connected NN
        fc_input = A{end};
        m = size(fc_input,4);
        fc_input = reshape(fc_input,numel(fc_input)/m,m);
        
        [A_fc,Z_fc] = forward_propagation(fc_input,W_fc,b_fc,activation);
        
        %% Compute cost
        cost(iter,k) = compute_cost(y_train,A_fc);
        %% Back prop propagates from last fully connected layer to last conv layer
        [dFC_input, dW_fc ,db_fc] = backward_propagation_fc_cnn(fc_input,y_train,A_fc,Z_fc,W_fc,activation);
        %% Last conv layer gradient is attained by reshaping the unfolded the parameters back to original shape
        dA{num_conv_layer} = reshape(dFC_input,size(A{end}));
        % Assuming this layer to be a max pool layer
        dZ{num_conv_layer} = dA{num_conv_layer};
        %% Back prop through the conv layers back to input image
        [dA, dZ, dFilt, dBias] = Conv_back_prop(dZ,Z,A,X,filt,s_x,s_y,s,conv_layer_name,max_mask);
        %% Update all parameters - SGD
        % FC layers
%         [W_fc,b_fc] = update_parameters(W_fc,b_fc,learning_rate,dW_fc,db_fc);
        % Conv layer
%         [filt,bias] = update_parameters_conv(filt,bias,learning_rate,dFilt,dBias,conv_layer_name);
        %% Update parameters - momentum
        [V_dW_fc, V_db_Fc] =  momentum_avg(V_dW_fc,dW_fc,V_db_fc,db_fc,iter);
        [W_fc,b_fc] = update_parameters(W_fc,b_fc,learning_rate,V_dW_fc,V_db_fc);
        
        [V_dFilt, V_dBias] =  momentum_avg(V_dFilt,dFilt,V_dBias,dBias,iter);
         [filt,bias] = update_parameters_conv(filt,bias,learning_rate,V_dFilt,V_dBias,conv_layer_name);


    end
end
%%
figure; 
plot(cost(:))
%% 

figure;
imagesc(y_train)

figure;
imagesc(A_fc{end})
