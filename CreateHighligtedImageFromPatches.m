% Bahri ABACI
clear all, close all, clc;
addpath(genpath('../data_analysis/functions'));
addpath(genpath('../data_analysis'));

SpatialResolution = 20;
MaxAllowedCloudPercentage = 0.01;

InputPath = 'E:\Dropbox\Dataset\satellite\sentinel2\';
LabeledDataFolder = 'data\labels';
MucilagePath = 'E:\Dropbox\Education\PhD\Projects\MucilageDetection\uNetLearning\outputs\35TPE\';

scaler = 10 ./ SpatialResolution;

TileNames = {'35TPE'};
TileCropZones = {[1,1, round(8276 * scaler), round(3096 * scaler)]};

% TileNames = {'35TPF'};
% TileCropZones = {[1,1, round(10980 * scaler), round(10980 * scaler)]};

MucilageIdentifier = sprintf('_%dm_MUCILAGE.mat',SpatialResolution);

% laod the water masks
WaterMasks = GetTileWaterMask(LabeledDataFolder, TileNames, SpatialResolution);

% make output directory
mkdir('model_outputs');

% get the all results
AllFiles = dir([MucilagePath, '*.mat']);

for f = 1:length(AllFiles)
    
    % set the tile index manually
    idx = 1;
    
    if ~contains(AllFiles(f).name, MucilageIdentifier)
        continue;
    end
    
    fileSubDir.folder = InputPath;
    fileSubDir.name = erase(AllFiles(f).name,MucilageIdentifier);
    fileSubDir.tile = TileNames{idx};
    
    % get the data
    SentinelData = GetSentinelData(fileSubDir, SpatialResolution);
    
    % get the true color image and water mask
    TCI = im2double(SentinelData.TCI);
    WaterMask = WaterMasks{idx} > 128;

    % check the cloud coverage and ignore too cloudy days
    WaterArea = sum(WaterMask(:));
    
    % get the cloud region
    if size(SentinelData.CloudMask, 1) > 0
        CloudOnWater = SentinelData.CloudMask & WaterMask;
    else
        CloudOnWater = zeros(size(WaterMask));
    end
    CloudOnWaterArea = sum(CloudOnWater(:));

    if (CloudOnWaterArea / WaterArea) >= MaxAllowedCloudPercentage
        fprintf('Skipping %s because its so cloudy!\n', fileSubDir.name);
        continue;
    end
    
    % load the file S2B_MSIL2A_20210514T084559_N0300_R107_T35TPE_20210514T113538_20m_MUCILAGE
    load(fullfile(AllFiles(f).folder, AllFiles(f).name));
    
    % create outputs
    TileCropZone = TileCropZones{idx};
    Prediction = zeros(size(TCI,1), size(TCI,2), 1);
    Prediction(TileCropZone(2):TileCropZone(4), TileCropZone(1):TileCropZone(3)) = mucilage;
    
    TCIOverlay = HighlightPredictionsOnImage(TCI, Prediction, WaterMask);
    TCIOverlay = imcrop(TCIOverlay, TileCropZone);
    
%     imshow(TCIOverlay, []);
%     drawnow;
%     
    FileName = SentinelFolderNameToFileName(AllFiles(f).name, sprintf('_%dm',SpatialResolution));
    imwrite(TCIOverlay, sprintf('model_outputs/%s.png', FileName));
end