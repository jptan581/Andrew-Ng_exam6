%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%% Initialization
clear ; close all; clc
%% ==================== Part 1: Email Preprocessing ====================
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');
%% ==================== Part 2: Feature Extraction ====================
fprintf('\nExtracting features from sample email (emailSample1.txt)\n');
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
features      = emailFeatures(word_indices);

fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

%% =========== Part 3: Train Linear SVM for Spam Classification ========
load('spamTrain.mat');
fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% =================== Part 4: Test Spam Classification ================
load('spamTest.mat');
p = svmPredict(model, Xtest);
%% ================= Part 5: Top Predictors of Spam ====================
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end
fprintf('\n\n');
%% =================== Part 6: Try Your Own Emails =====================
filename = 'spamSample1.txt';
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);
fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

