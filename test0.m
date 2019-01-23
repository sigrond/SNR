leafDatasetPath = fullfile('Folio Leaf Dataset','Folio');
imds = imageDatastore(leafDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

tbl = countEachLabel(imds)

[trainingSet,validationSet, testSet] = splitEachLabel(imds, 0.5, 0.25, 0.25, 'randomized');

layers = [
    imageInputLayer([227 227 3],'Name','Image Input 227x227x3')
    %warstwa 1
    convolution2dLayer(11,96,'Stride',4,'Padding',0,'Name','Convolution 96 11x11 stride [4  4] padding [0  0  0  0]')
    reluLayer('Name','1. ReLU')
    maxPooling2dLayer(3,'Stride',2,'Name','1. Max Pooling 3x3 stride [2  2] padding [0  0  0  0]')
    %warstwa 2
    crossChannelNormalizationLayer(5,'K',1,'Name','2. Cross Channel Normalization with 5 channels per element')
    convolution2dLayer(5,256,'Stride',1,'Padding',2,'Name','Convolution 256 5x5 stride [1  1] padding [2  2  2  2]')
    reluLayer('Name','2. ReLU')
    maxPooling2dLayer(3,'Stride',2,'Name','2. Max Pooling 3x3 stride [2  2] padding [0  0  0  0]')
    %warstwa 3
    crossChannelNormalizationLayer(5,'K',1,'Name','3. Cross Channel Normalization with 5 channels per element')
    convolution2dLayer(3,384,'Stride',1,'Padding',1,'Name','3. Convolution 384 3x3 stride [1  1] padding [1  1  1  1]')
    reluLayer('Name','3. ReLU')
    %warstwa 4
    convolution2dLayer(3,384,'Stride',1,'Padding',1,'Name','4. Convolution 384 3x3 stride [1  1] padding [1  1  1  1]')
    reluLayer('Name','4. ReLU')
    %warstwa 5
    convolution2dLayer(3,256,'Stride',1,'Padding',1,'Name','Convolution 256 3x3 stride [1  1] padding [1  1  1  1]')
    reluLayer('Name','5. ReLU')
    maxPooling2dLayer(3,'Stride',2,'Name','5. Max Pooling 3x3 stride [2  2] padding [0  0  0  0]')
    %warstwa 6
    fullyConnectedLayer(4096,'Name','6. Fully Connected 4096 layer')
    reluLayer('Name','6. ReLU')
    dropoutLayer('Name','6. Dropout 50%')
    %warstwa 7
    fullyConnectedLayer(4096,'Name','7. Fully Connected 4096 layer')
    reluLayer('Name','7. ReLU')
    dropoutLayer('Name','7. Dropout 50%')
    %warstwa 8
    fullyConnectedLayer(32,'Name','Fully Connected 32 layer')
    softmaxLayer('Name','Softmax')
    classificationLayer('Name','Classification Output crossentropyex')
 ]
lgraph = layerGraph(layers)

imageSize = [227 227 3];
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedValidationSet = augmentedImageDatastore(imageSize, validationSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.00000000000000001, ... %to tak jakby nie by³o uczenia
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedValidationSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

load 'myAlexNet0.mat';
if ~exist('net', 'var')
    net = trainNetwork(augmentedTrainingSet,layers,options);
end

[YPred,scores] = classify(net,augmentedTestSet);
[S,I] = maxk(scores',5);
YValidation = testSet.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

top5 = sum(sum(tbl.Label(I)' == YValidation))/numel(YValidation)
