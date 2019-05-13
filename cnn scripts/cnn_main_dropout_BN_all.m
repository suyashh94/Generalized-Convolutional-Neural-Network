
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

%% Initialising momentum terms for filters
for i = 1:num_conv_layer
    if conv_layer_name(i) == 1
        V_dFilt{i} = zeros(size(filt{i}));
    else
        V_dFilt{i} = 0;
    end
end

%% Output shape of CNN layers
[output_shape] = cnnOutputShape(x_train,conv_layer_name,n_f,n_c,s_x,s_y,s);

%% Initializing gamma and beta for every layer
for i = 1:num_conv_layer
    gamma{i} = ones(output_shape(1,i)*output_shape(2,i)*output_shape(3,i),1);
    beta{i} = zeros(output_shape(1,i)*output_shape(2,i)*output_shape(3,i),1);
end
%% Initializing momentum term for gamma and beta

for i = 1:num_conv_layer
    V_dGamma{i} = zeros(size(gamma{i}));
end

for i = 1:num_conv_layer
    V_dBeta{i} = zeros(size(beta{i}));
end

%% Initializing mu_test and sigma_test for conv layers
for i = 1:num_conv_layer
    mu_test_cnn{i} = zeros(size(gamma{i}));
    sigma_test_cnn{i} = zeros(size(gamma{i}));
end
%% keep_prob threshold for dropout in CNN layers
keep_prob = 0.8;

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
[W_fc, ~] = initialize_parameters(layer_dims);
[gamma_fc , beta_fc] = initialize_gamma_and_beta(layer_dims);
%% Initializing mu_test and sigma_test for fc layer
for i = 1:num_layers
    mu_test_fc{i} = zeros(layer_dims(i+1),1);
    sigma_test_fc{i} = zeros(layer_dims(i+1),1);
end
%% Initialize V_dw and , V_dGamma, V_dBeta for gradient descent with momentum
[V_dW_fc, ~] = initialize_parameters_with_zero(layer_dims);
[V_dGamma_fc, V_dBeta_fc] = initialize_parameters_with_zero_beta_gamma(layer_dims);
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

learning_rate = 0.5;
count = 0;
keep_prob_fc = 0.8;
keep_prob_init_fc = 1;
%% Iteration starts
for iter = 1:100
    for k = 1:length(X_minibatch)
        k
        count = count + 1;
        disp(['iteration number ', num2str(iter)])
        X = X_minibatch{k};
        y_train = y_train_minibatch{k};
        %% keepMatrix for every layer after dropout
        toKeep_matrix = getDropout_keepMatrix(conv_layer_name,output_shape,minibatch_size,keep_prob);
        %% Forward propragation for convolution layers
        [Z,A,Z_bn_unfolded,Z_hat_unfolded,gamma,...
            Z_mu_unfolded, ivar, var_sqrt, var, eps,max_mask] = ...
            Conv_forward_prop_dropout_BN(X,conv_layer_name,filt,s_x,s_y,s,toKeep_matrix,keep_prob,gamma,beta);
        
        for layer = 1:num_conv_layer
            Z_unfolded{layer} = reshape(Z{layer},numel(Z{layer})/minibatch_size, minibatch_size);
        end
        % Weighted average for mu and var for test
        [mu_test_cnn,sigma_test_cnn] = moving_avg_BN(Z_unfolded,Z_mu_unfolded,var,mu_test_cnn,...
            sigma_test_cnn);
        
        %% Unfold the final conv layer out and forward prop using a fully connected NN
        fc_input = A{end};
        m = size(fc_input,4);
        fc_input = reshape(fc_input,numel(fc_input)/m,m);
        %% Making prob_matrix for FC layers
        for layer = 1:length(layer_dims)
            prob_matrix_fc{layer} = rand(layer_dims(layer), minibatch_size);
            if layer == 1
                prob_matrix_fc{layer} = prob_matrix_fc{layer} < keep_prob_init_fc;
            elseif layer == length(layer_dims)
                prob_matrix_fc{layer} = prob_matrix_fc{layer} > -1;    % Everything equal to 1 for output layer
            else
                prob_matrix_fc{layer} = prob_matrix_fc{layer} < keep_prob_fc;
            end
        end
        %% Forward prop through FC layer
        [A_fc,Z_fc,Z_bn_fc,Z_hat_fc,gamma_fc, Z_mu_fc, ivar_fc, var_sqrt_fc, var_fc, eps_fc] ...
            = forward_propagation_drop_out_BN(fc_input,W_fc,gamma_fc,beta_fc,...
            activation,prob_matrix_fc,keep_prob_init_fc,keep_prob_fc)  ;
        
        % Weighted average for mu and var for test
        [mu_test_fc,sigma_test_fc] = moving_avg_BN(Z_fc,Z_mu_fc,var_fc,mu_test_fc,sigma_test_fc);
        
        %% Compute cost
        cost(iter,k) = compute_cost(y_train,A_fc);
        %% Back prop propagates from last fully connected layer to last conv layer
        
        [dFC_input ,dW_fc ,d_gamma_fc, d_beta_fc] = backward_propagation_drop_out_BN_cnn...
            (fc_input,y_train,A_fc,Z_bn_fc,Z_hat_fc,gamma_fc,Z_mu_fc,ivar_fc,var_sqrt_fc...
            ,var_fc,eps_fc,W_fc,activation,prob_matrix_fc,keep_prob_init_fc,keep_prob_fc);
        
        
        %% Last conv layer gradient is attained by reshaping the unfolded the parameters back to original shape
        dA{num_conv_layer} = reshape(dFC_input,size(A{end}));
        % Assuming this layer to be a max pool layer
        [dZ{num_conv_layer},dGamma_end,dBeta_end] = ...
            batch_norm_backward_maxpool(dA{num_conv_layer},Z_hat_unfolded{num_conv_layer},...
            gamma{num_conv_layer}, Z_mu_unfolded{num_conv_layer}, ivar{num_conv_layer}, ...
            var_sqrt{num_conv_layer}, var{num_conv_layer}, eps{num_conv_layer});
        dFilt{num_conv_layer} = 0;
        %% Back prop through the conv layers back to input image
        [dA, dZ, dFilt, dGamma, dBeta] = ...
            Conv_back_prop_dropout_BNN...
            (dZ,A,X,filt,s_x,s_y,s,conv_layer_name,max_mask,toKeep_matrix,keep_prob...
            ,Z_bn_unfolded,Z_hat_unfolded,Z_mu_unfolded,gamma,...
            ivar, var_sqrt, var, eps)    ;
        
        dGamma{num_conv_layer} = dGamma_end;
        dBeta{num_conv_layer} = dBeta_end;
        %% Update all parameters - SGD
        % FC layers
        %         [W_fc,b_fc] = update_parameters(W_fc,b_fc,learning_rate,dW_fc,db_fc);
        % Conv layer
        %         [filt,bias] = update_parameters_conv(filt,bias,learning_rate,dFilt,dBias,conv_layer_name);
        %% Update parameters - momentum
        [V_dW_fc, V_dGamma_fc, V_dBeta_fc] = momentum_avg_BN(V_dW_fc,dW_fc,V_dGamma_fc,...
            d_gamma_fc,V_dBeta_fc,d_beta_fc,count);
        [W_fc,gamma_fc,beta_fc] = update_parameters_BN(W_fc,gamma_fc,beta_fc,...
            learning_rate,V_dW_fc,V_dGamma_fc,V_dBeta_fc);
        
        [V_dFilt, V_dGamma, V_dBeta] = momentum_avg_BN(V_dFilt,dFilt,V_dGamma,dGamma,V_dBeta,dBeta,count);
        [filt,gamma,beta] = update_parameters_conv_BN(filt,gamma,beta,learning_rate,V_dFilt,V_dGamma,V_dBeta);
        
        
    end
end
%% Plot the cost
figure;
plot(cost(:))
%%

figure;
imagesc(y_train)

figure;
imagesc(A_fc{end})

%% Testing

% Define test set - first 100 samples of training set here. Can be set to
% test set data
x_test = x_train(:,:,:,1:100);
y_test =  y(:,1:100);

%% Forward prop through CNN

[Z_test,A_test,Z_bn_unfolded_test,Z_hat_unfolded_test,gamma_test,...
    Z_mu_unfolded_test, ivar_test, var_sqrt_test, var_test, eps_test,max_mask_test] =...
    Conv_forward_prop_BN_Test...
    (x_test,conv_layer_name,filt,s_x,s_y,s,gamma,beta,mu_test_cnn,sigma_test_cnn);
%% Unfolding 
fc_input_test = A_test{end};
m_test = size(fc_input_test,4);
fc_input_test = reshape(fc_input_test,numel(fc_input_test)/m_test,m_test);

%% Forward prop through fc layers
[A_main,~] = forward_propagation_BN_Test(fc_input_test,W_fc,gamma_fc,beta_fc,activation,mu_test_fc,sigma_test_fc);

A_predict = A_main{end};
y_predict = zeros(size(A_predict));
for i = 1:size(A_predict,2)
    [~,dummy] = max(A_predict(:,i));
    y_predict(dummy,i) = 1;
end
%%

figure;
hold on
subplot(2,1,1)
imagesc(y_test)

subplot(2,1,2)
imagesc(y_predict)

%% error

error = (sum(abs(y_test - y_predict)));
error = length(find(error > 0))/ size(y_test,2);

%% Confusion Matrix
for i = 1:size(y_test,2)
    y_predict_val(i) = find(y_predict(:,i) == 1);
end

for i = 1:size(y_test,2)
    y_test_val(i) = find(y_test(:,i) == 1);
end

C = confusionmat(y_test_val, y_predict_val);

figure;
imagesc(C);



