% main code
% train network

clear all;
close all;
%% 1. Variable Setting
% PosPatch: image of face landmark
% NegPatch: image of NOT face landmark
numImage = 811;
numPosPatch = 25;
numNegPatch = 120;

%% 2. Training
% for 42 Landmarks
% 1 ~ 9 : Contour Landmark -> eliminate
for i = 10 : 42
    [tmpAccuracy, tmpNet] = LandmarkEvaluator(i, numImage, numPosPatch, numNegPatch);
    lm_accuracy{i-9} = tmpAccuracy;
    lm_net{i-9} = tmpNet;
end

%% 3. Save Variable
save network_rate lm_accuracy
save network_set lm_net