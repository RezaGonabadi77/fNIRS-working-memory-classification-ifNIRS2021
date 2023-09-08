clc
clear
close all
%% Load Data
load('sub03.mat')
% plot(DataTrain_Hb(:,15,7))
%% step 1-0: channel selection
dataTest1 = DataTrain_HbO2(:,:,:);
dataTest2 = DataTrain_Hb(:,:,:);
% dataTest = dataTest1 + dataTest2;
plot(dataTest1(:,1,2))
hold on
plot(dataTest2(:,1,2))
grid on
%% step 1:  cut Labels from DataTrain
Block_Train= ClassLabel_Train;
% HbO2
c1=0;
c2=0;
c3=0;
for i=1:numel(Block_Train)
    if Block_Train(i) ==1
        c1=c1+1;
        data1(:,:,c1)=dataTest(:,:,i);
    elseif Block_Train(i)== 2
        c2=c2+1;
        data2(:,:,c2)=dataTest(:,:,i);
    elseif Block_Train(i)== 3
        c3=c3+1;
        data3(:,:,c3)=dataTest(:,:,i);
    end
end
% % Hb
% c1=0;
% c2=0;
% c3=0;
% for i=1:numel(Block_Train)
%     if Block_Train(i) ==1
%         c1=c1+1;
%         data1Hb(:,:,c1)=DataTrain_Hb(:,sel(1:numChn),i);
%     elseif Block_Train(i)== 2
%         c2=c2+1;
%         data2Hb(:,:,c2)=DataTrain_Hb(:,sel(1:numChn),i);
%     elseif Block_Train(i)== 3
%         c3=c3+1;
%         data3Hb(:,:,c3)=DataTrain_Hb(:,sel(1:numChn),i);
%     end
% end
%% Concat 3D matrix
% data1 = cat(3,data1Hb,data1HbO2);
% data2 = cat(3,data2Hb,data2HbO2);
% data3 = cat(3,data3Hb,data3HbO2);
%% step 2-1: Feature Extaction Time Domain
for i=1:size(data1,3)
    X1= data1(:,:,i);
    X2= data2(:,:,i);
    X3= data3(:,:,i);
    for j=1:size(X1,2)
        tp1(:,j)= myfeatureExtraction(X1(:,j));
        tp2(:,j)= myfeatureExtraction(X2(:,j));
        tp3(:,j)= myfeatureExtraction(X3(:,j));
    end
    Features1(:,i)= tp1(:);
    Features2(:,i)= tp2(:);
    Features3(:,i)= tp3(:);
end
data1Time = Features1;
data2Time = Features2;
data3Time = Features3;
for i=1:size(dataTest,3)
    X1= dataTest(:,:,i);
    for j=1:size(X1,2)
        tp1(:,j)= myfeatureExtraction(X1(:,j));
    end
    FeaturesTest(:,i)= tp1(:);
end
dataTestTime = FeaturesTest;
%% step 2-2: Feature Extaction Frequency Domain
c1 = 1;
for i=1:size(data1,3)
    sigA= data1(:,:,i);
    sigB= data2(:,:,i);
    sigC= data3(:,:,i);
    % feature extraction
    % fourier transform
    NA= length(sigA);
    NB= length(sigB);
    NC= length(sigC);
    fx_a= abs(fft2(sigA));
    fx_b= abs(fft2(sigB));
    fx_c= abs(fft2(sigC));
    % select half of coeficients
    fx_a= fx_a(1:round(NA/2),:);
    fx_b= fx_b(1:round(NB/2),:);
    fx_c= fx_c(1:round(NC/2),:);
    % calculate Magnitude of coeficients
    data1FreqTrain(:,:,c1) = fx_a;
    data2FreqTrain(:,:,c1) = fx_b;
    data3FreqTrain(:,:,c1) = fx_c;
    c1 = c1 + 1;
end

for i=1:size(data1FreqTrain,3)
    X1= data1FreqTrain(:,:,i);
    X2= data2FreqTrain(:,:,i);
    X3= data3FreqTrain(:,:,i);
    for j=1:size(X1,2)
        tp1Freq(:,j)= myfeatureExtraction(X1(:,j));
        tp2Freq(:,j)= myfeatureExtraction(X2(:,j));
        tp3Freq(:,j)= myfeatureExtraction(X3(:,j));
    end
    Features1Freq(:,i)= tp1Freq(:);
    Features2Freq(:,i)= tp2Freq(:);
    Features3Freq(:,i)= tp3Freq(:);
end
data1Freq = Features1Freq;
data2Freq = Features2Freq;
data3Freq = Features3Freq;
% Data Test
c1 = 1;
for i=1:size(dataTest,3)
    sigA= dataTest(:,:,i);
    % feature extraction
    % fourier transform
    NA= length(sigA);
    fx_a= abs(fft2(sigA));
    % select half of coeficients
    fx_a= fx_a(1:round(NA/2),:);
    % calculate Magnitude of coeficients
    dataTestFreqTrain(:,:,c1) = fx_a;
    c1 = c1 + 1;
end

for i=1:size(dataTestFreqTrain,3)
    X1= dataTestFreqTrain(:,:,i);
    for j=1:size(X1,2)
        tp1Test(:,j)= myfeatureExtraction(X1(:,j));
    end
    Features1FreqTest(:,i)= tp1Test(:);
end
dataTestFreq = Features1FreqTest;
%% Features Selection By PCA
data1 = [data1Time ; data1Freq];
data2 = [data2Time ; data2Freq];
data3 = [data3Time ; data3Freq];
DataTest = [dataTestTime ; dataTestFreq];
data = [data1 data2 data3 DataTest];
% data = [data1 data2 data3];
% d = 20; % #Dimension or #Features
% w = myPCA(data, d);
% f1 = w' * data;
% 
% data1 = f1(:,1:8);
% data2 = f1(:,9:16);
% data3 = f1(:,17:24);
% DataTest = f1(:,25:36);

% 
% data1 = repmat(data1,1,2);
% data2 = repmat(data2,1,2);
% data3 = repmat(data3,1,2);

% cols = size(data1,2);
% P = randperm(cols);
% data1N = data1(:,P);
% data2N = data2(:,P);
% data3N = data3(:,P);
% data1 = [data1 , data1N];
% data2 = [data2 , data2N];
% data3 = [data3 , data3N];

%% normalization
% data = [data1 , data2 , data3];
% mu= mean(data,2);
% sigma= std(data')';
% data= (data- repmat(mu,1,size(data,2)))./repmat(sigma,1,size(data,2)) ;
% 
% data1 = data(:,1:8);
% data2 = data(:,9:16);
% data3 = data(:,17:24);

%%  classification
% step 6-1: devide data into train and test data
% k-fold cross validation(k=5)
k=8;
fold1= floor(size(data1,2) / k);
fold2= floor(size(data2,2) / k);
fold3= floor(size(data3,2) / k);
Ct=0;
C1=1;
for i=1:k
    indxtest1= (i-1)*fold1+1:i*fold1;
    indxtrain1=1:size(data1,2);
    indxtrain1(indxtest1)=[];
    
    indxtest2= (i-1)*fold2+1:i*fold2;
    indxtrain2=1:size(data2,2);
    indxtrain2(indxtest2)=[];
    
        
    indxtest3= (i-1)*fold3+1:i*fold3;
    indxtrain3=1:size(data3,2);
    indxtrain3(indxtest3)=[];
    
    traindata1= data1(:,indxtrain1);
    testdata1= data1(:,indxtest1);
    %
    traindata2= data2(:,indxtrain2);
    testdata2= data2(:,indxtest2);
    
    traindata3= data3(:,indxtrain3);
    testdata3= data3(:,indxtest3);
    %
    traindata= [traindata1,traindata2,traindata3];
    trainlabel= [ones(1,size(traindata1,2)),2*ones(1,size(traindata2,2)),...
        3*ones(1,size(traindata3,2))];
    
    testdata= [testdata1,testdata2,testdata3];
    testlabel= [ones(1,size(testdata1,2)),2*ones(1,size(testdata2,2)),...
        3*ones(1,size(testdata3,2))];
    %% step 2: train model using train data and train label
%     mdl= fitcknn(traindata',trainlabel,'NumNeighbors',5);
%     mdl= fitcdiscr(traindata',trainlabel);
%     mdl= fitctree(traindata',trainlabel);
%     kernel='linear';
%     mdl = mymultisvmtrainOVO(traindata,trainlabel,kernel);
        mdl= fitcnb(traindata',trainlabel);
%         mdl = mySLPtrain(traindata,trainlabel);
    %% step 3: test trained model using test data
%     output= mymultisvmclassifyOVO(mdl,testdata);
    userlabel=unique(trainlabel);
    outputTest(C1 , :)= predict(mdl,DataTest');
    C1 = C1 + 1;
    output = predict(mdl,testdata');
%     output = mySLPtest(mdl,testdata);
    %% step 4: validation
    C= confusionmat(testlabel,output);
    Ct= Ct+C;
    accuracy(i)= sum(diag(C)) / sum(C(:))*100;
    accuracy1(i)= C(1,1) / sum(C(1,:))*100;
    accuracy2(i)= C(2,2) / sum(C(2,:))*100;
    accuracy3(i)= C(3,3) / sum(C(3,:))*100;
end
Ct
disp(['Total Accuracy: ',num2str(mean(accuracy)),'%'])
disp(['Accuracy 1: ',num2str(mean(accuracy1)),'%'])
disp(['Accuracy 2: ',num2str(mean(accuracy2)),'%'])
disp(['Accuracy 3: ',num2str(mean(accuracy3)),'%'])
Total = myMejorityVoting(outputTest,userlabel);