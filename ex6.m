%% Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%% Initialization
clear ; close all; clc
%% =============== Part 1: Loading and Visualizing Data ================
load('ex6data1.mat');
plotData(X, y);
%% ==================== Part 2: Training Linear SVM ====================
load('ex6data1.mat');
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
%% =============== Part 3: Implementing Gaussian Kernel ===============
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);
%% =============== Part 4: Visualizing Dataset 2 ================
fprintf('Loading and Visualizing Data ...\n')
load('ex6data2.mat');
plotData(X, y);
%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
load('ex6data2.mat');
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);
%% =============== Part 6: Visualizing Dataset 3 ================
fprintf('Loading and Visualizing Data ...\n')
load('ex6data3.mat');
plotData(X, y);
%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
load('ex6data3.mat');
[C, sigma] = dataset3Params(X, y, Xval, yval);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

