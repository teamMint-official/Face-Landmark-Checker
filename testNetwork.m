% network test code

clear all;
close all;

%% 1. Input Setting
load('network_set.mat');
testImage = imread('TestSet/1.jpg');
testLandmark = load('TestSet/1.mat');

%% 2. Landmark Selection
selectLandmark = [];
rates = [];
patch = zeros(25, 25, 3, 33);

% evaluate inner landmark only
for i = 10:42
    network = lm_net{i-9};
    marker = testLandmark(i,:);

    tmpPatch = imcrop(testImage, [marker - [12,12] 24 24]);
    patch(:,:,:,i-9) = tmpPatch;

    [result, rate] = classify(network, patch(:, :, :, i-9));

    if rate(1,2) > 0.3
        selectLandmark = [selectLandmark, i];
        rates = [rates, rate(1,2)];
    end
    name{i-9} = round(100 * rate(1,2));
end

%% 3. Visualization
figure, imshow(testImage);
hold on
scatter(testLandmark(10:42,1),testLandmark(10:42,2));
scatter(testLandmark(selectLandmark,1),testLandmark(selectLandmark,2), 'filled');
text(testLandmark(10:42,1), testLandmark(10:42,2), name);
hold off