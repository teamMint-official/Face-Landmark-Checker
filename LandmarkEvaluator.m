function [accuracy, net] = LandmarkEvaluator(lmidx, numImage, numPosPatch ,numNegPatch)
% Training the Landmark Evaluator
% lmidx: landmark index, 1~9: contour landmark, 10~42: evaluation target
% PosPatch: image of face landmark
% NegPatch: image of NOT face landmark


%% 1. Classify DataSet
% Classify Color & Gray Image in Training Set
numColorImage = numImage;
for i = 1:numImage
    strImages = ['TrainingSet/' num2str(i) '.jpg'];
    tmpImages = imread(strImages);
    if length(tmpImages(1, 1, :)) < 3
        numColorImage = numColorImage - 1;
    end
end

%% 2. Variable Setting
numValidationImage = 200;
numTrainingImage = numColorImage - 200;
numAllPatch = numPosPatch + numNegPatch;
keypoints = load('TrainingSet/keypoints.txt');
keypoints = floor(keypoints);

%% 3. Patch Setting
% NegPatch = -1, PosPatch = 1

% Initialize the Train Patch
trainPatches = zeros(25, 25, 3, numTrainingImage * numAllPatch);
trainLabels = -1 * ones(numTrainingImage * numAllPatch, 1);
for i = 1:numPosPatch
    trainLabels(i : numAllPatch : (numTrainingImage - 1) * numAllPatch + i, 1) = 1;
end
trainLabels = categorical(trainLabels);

% Initialize the Validation Patch
valPatches = zeros(25, 25, 3, numValidationImage * numAllPatch);
valLabels = -1 * ones(numValidationImage*numAllPatch, 1);
for i = 1:numPosPatch
    valLabels(i : numAllPatch : (numValidationImage - 1) * numAllPatch + i, 1) = 1;
end
valLabels = categorical(valLabels);

% Patch maker
pixelWindows = repmat(-2:2, 5, 1);
pixelWindows = pixelWindows(:);
pixelWindows(:,2) = repmat((-2:2)',5,1);
fillValidation = 0;
for i = 1 : numImage
    keypoint = keypoints(i, :);
    marker = keypoint(1, 2 * lmidx - 1 : 2 * lmidx);
    strImages = ['TrainingSet/' num2str(i) '.jpg'];
    tmpImages = imread(strImages);
    tmpImages = im2double(tmpImages);
    if size(tmpImages, 3) == 3
        width = size(tmpImages, 2);
        height = size(tmpImages, 1);
        if fillValidation < 200
            fillValidation = fillValidation + 1;
            for j = 1 : numPosPatch
                tmpPatch = imcrop(tmpImages, [marker - pixelWindows(j, :) - [12, 12] 24 24]);
                if(size(tmpPatch, 2) < 25 || size(tmpPatch, 1) < 25)
                    tmpPatch = imresize(tmpPatch, [25 25]);
                end
                valPatches(:, :, :, (fillValidation - 1) * numAllPatch + j) = tmpPatch;
            end
            for j = 1:numNegPatch
                xShift = randperm(width - 25, 1);
                if xShift > marker(1, 1) - 10 && xShift < marker(1, 1) - 4
                    xShift = xShift + 5;
                end
                yShift = randperm(height - 25,1);
                if yShift > marker(1, 1) - 10 && yShift < marker(1, 1) - 4
                    yShift = yShift + 5;
                end
                tmpPatch = imcrop(tmpImages, [xShift yShift 24 24]);
                if(size(tmpPatch, 2) < 25 || size(tmpPatch, 1) < 25)
                    tmpPatch = imresize(tmpPatch, [25 25]);
                end
                valPatches(:, :, :, (fillValidation - 1) * numAllPatch + numPosPatch + j) = tmpPatch;
            end
        else
            fillValidation = fillValidation+1;
            for j = 1:numPosPatch
                tmpPatch = imcrop(tmpImages, [marker - pixelWindows(j, :) - [12, 12] 24 24]);
                if(size(tmpPatch, 2) < 25 || size(tmpPatch, 1) < 25)
                    tmpPatch = imresize(tmpPatch, [25 25]);
                end
                trainPatches(:, :, :, (fillValidation - 201) * numAllPatch + j) = tmpPatch;
            end
            for j = 1 : numNegPatch
                xShift = randperm(width - 25, 1);
                if xShift > marker(1, 1) - 10 && xShift < marker(1, 1) - 4
                    xShift = xShift + 5;
                end
                yShift = randperm(height - 25, 1);
                if yShift > marker(1, 1) - 10 && yShift < marker(1, 1) - 4
                    yShift = yShift + 5;
                end
                tmpPatch = imcrop(tmpImages, [xShift yShift 24 24]);
                if(size(tmpPatch, 2) < 25 || size(tmpPatch, 1) < 25)
                    tmpPatch = imresize(tmpPatch, [25 25]);
                end
                trainPatches(:, :, :, (fillValidation - 201) * numAllPatch + numPosPatch + j) = tmpPatch;
            end
        end  
    end
end

%% 4. Network Definition
layers = [
    imageInputLayer([25 25 3])

    convolution2dLayer(5, 6, 'Padding', 1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2)

    convolution2dLayer(3, 16, 'Padding', 1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(100)
    fullyConnectedLayer(40)
    fullyConnectedLayer(2)

    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
    ];

options = trainingOptions('sgdm',...
    'ValidationData',{valPatches,valLabels},...
    'Verbose',false,...
    'Plots','training-progress');

%% 5. Network Training
net = trainNetwork(trainPatches, trainLabels, layers, options);
test_result = classify(net, valPatches);
accuracy = sum(test_result == valLabels) / length(valLabels);