clc
clear
close all
%% Load Data
load('sub15_Test_3_Competition_2021.mat')
% plot(DataTrain_Hb(:,15,7))
% %% Filter Design
% fs = 10;  
% fl = 0.15; 
% fh = 4.99; 
% wn = [fl fh]/(fs/2); 
% order =3;
% [b,a] = butter(order,wn,'bandpass'); 
% %% Filtering Data
% % Train Data
% trials =24;
% for i =1:trials 
%     for j =1:24
%      x1 =  DataTrain_Hb(:,j,i); 
%      x2 =  DataTrain_HbO2(:,j,i); 
%      DataTrain_Hb(:,j,i) = filtfilt(b,a,x1); 
%      DataTrain_HbO2(:,j,i) = filtfilt(b,a,x2);
%     end 
% end
% trials =12;
% for i =1:trials 
%     for j =1:24
%      x1 =  DataTest_3_Hb(:,j,i); 
%      x2 =  DataTest_3_HbO2(:,j,i); 
%      DataTest_3_Hb(:,j,i) = filtfilt(b,a,x1); 
%      DataTest_3_HbO2(:,j,i) = filtfilt(b,a,x2);
%     end 
% end



%% step 1-0: channel selection
load('indxChn_anova_g.mat')
numChn= 15;
% sel(1:numChn)
dataTest_HbO2 = DataTest_3_HbO2(70:470,sel(1:numChn),:);
dataTest_Hb = DataTest_3_Hb(70:470,sel(1:numChn),:);
%% step 1:  cut Labels from DataTrain
Block_Train= ClassLabel_Train;
% HbO2
c1=0;
c2=0;
c3=0;
for i=1:numel(Block_Train)
    if Block_Train(i) ==1
        c1=c1+1;
        data1_HbO2(:,:,c1)=DataTrain_HbO2(70:470,sel(1:numChn),i);
        data1_Hb(:,:,c1)=DataTrain_Hb(70:470,sel(1:numChn),i);
    elseif Block_Train(i)== 2
        c2=c2+1;
        data2_HbO2(:,:,c2)=DataTrain_HbO2(70:470,sel(1:numChn),i);
        data2_Hb(:,:,c2)=DataTrain_Hb(70:470,sel(1:numChn),i);
    elseif Block_Train(i)== 3
        c3=c3+1;
        data3_HbO2(:,:,c3)=DataTrain_HbO2(70:470,sel(1:numChn),i);
        data3_Hb(:,:,c3)=DataTrain_Hb(70:470,sel(1:numChn),i);
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
for i=1:size(data1_HbO2,3)
    X11= data1_HbO2(:,:,i);
    X12= data1_Hb(:,:,i);
    X21= data2_HbO2(:,:,i);
    X22= data2_Hb(:,:,i);
    X31= data3_HbO2(:,:,i);
    X32= data3_Hb(:,:,i);
    for j=1:size(X11,2)
        tp11(:,j)= myfeatureExtraction(X11(:,j));
        tp12(:,j)= myfeatureExtraction(X12(:,j));
        tp21(:,j)= myfeatureExtraction(X21(:,j));
        tp22(:,j)= myfeatureExtraction(X22(:,j));
        tp31(:,j)= myfeatureExtraction(X31(:,j));
        tp32(:,j)= myfeatureExtraction(X32(:,j));
    end
    Features1(:,i)= [tp11(:) ; tp12(:)];
    Features2(:,i)= [tp21(:) ; tp22(:)];
    Features3(:,i)= [tp31(:) ; tp32(:)];
end
data1Time = Features1;
data2Time = Features2;
data3Time = Features3;
for i=1:size(dataTest_HbO2,3)
    X1= dataTest_HbO2(:,:,i);
    X2= dataTest_Hb(:,:,i);
    for j=1:size(X1,2)
        tp1(:,j)= myfeatureExtraction(X1(:,j));
        tp2(:,j)= myfeatureExtraction(X2(:,j));
    end
    FeaturesTest(:,i)= [tp1(:) ; tp2(:)];
end
dataTestTime = FeaturesTest;
%% step 2-2: Feature Extaction Frequency Domain
c1 = 1;
for i=1:size(data1_HbO2,3)
    sigA1= data1_HbO2(:,:,i);
    sigB1= data2_HbO2(:,:,i);
    sigC1= data3_HbO2(:,:,i);
    sigA2= data1_Hb(:,:,i);
    sigB2= data2_Hb(:,:,i);
    sigC2= data3_Hb(:,:,i);
    % feature extraction
    % fourier transform
    NA1= length(sigA1);
    NB1= length(sigB1);
    NC1= length(sigC1);
    NA2= length(sigA2);
    NB2= length(sigB2);
    NC2= length(sigC2);
    fx_a1= abs(fft2(sigA1));
    fx_b1= abs(fft2(sigB1));
    fx_c1= abs(fft2(sigC1));
    fx_a2= abs(fft2(sigA2));
    fx_b2= abs(fft2(sigB2));
    fx_c2= abs(fft2(sigC2));
    % select half of coeficients
    fx_a1= fx_a1(1:round(NA1/2),:);
    fx_b1= fx_b1(1:round(NB1/2),:);
    fx_c1= fx_c1(1:round(NC1/2),:);
    fx_a2= fx_a1(1:round(NA2/2),:);
    fx_b2= fx_b1(1:round(NB2/2),:);
    fx_c2= fx_c1(1:round(NC2/2),:);
    % calculate Magnitude of coeficients
    data1FreqTrain1(:,:,c1) = fx_a1;
    data2FreqTrain1(:,:,c1) = fx_b1;
    data3FreqTrain1(:,:,c1) = fx_c1;
    data1FreqTrain2(:,:,c1) = fx_a2;
    data2FreqTrain2(:,:,c1) = fx_b2;
    data3FreqTrain2(:,:,c1) = fx_c2;
    c1 = c1 + 1;
end

for i=1:size(data1FreqTrain1,3)
    X11= data1FreqTrain1(:,:,i);
    X21= data2FreqTrain1(:,:,i);
    X31= data3FreqTrain1(:,:,i);
    X12= data1FreqTrain2(:,:,i);
    X22= data2FreqTrain2(:,:,i);
    X32= data3FreqTrain2(:,:,i);
    for j=1:size(X1,2)
        tp1Freq1(:,j)= myfeatureExtraction(X11(:,j));
        tp2Freq1(:,j)= myfeatureExtraction(X21(:,j));
        tp3Freq1(:,j)= myfeatureExtraction(X31(:,j));
        tp1Freq2(:,j)= myfeatureExtraction(X12(:,j));
        tp2Freq2(:,j)= myfeatureExtraction(X22(:,j));
        tp3Freq2(:,j)= myfeatureExtraction(X32(:,j));
    end
    Features1Freq(:,i)= [tp1Freq1(:) ; tp1Freq2(:)];
    Features2Freq(:,i)= [tp2Freq1(:) ; tp2Freq2(:)];
    Features3Freq(:,i)= [tp3Freq1(:) ; tp3Freq2(:)];
end
data1Freq = Features1Freq;
data2Freq = Features2Freq;
data3Freq = Features3Freq;
% Data Test
c1 = 1;
for i=1:size(dataTest_HbO2,3)
    sigA= dataTest_HbO2(:,:,i);
    sigB= dataTest_Hb(:,:,i);
    % feature extraction
    % fourier transform
    NA= length(sigA);
    NB= length(sigB);
    fx_a1= abs(fft2(sigA));
    fx_a2= abs(fft2(sigB));
    % select half of coeficients
    fx_a1= fx_a1(1:round(NA/2),:);
    fx_a2= fx_a2(1:round(NB/2),:);
    % calculate Magnitude of coeficients
    dataTestFreqTrain1(:,:,c1) = fx_a1;
    dataTestFreqTrain2(:,:,c1) = fx_a2;
    c1 = c1 + 1;
end

for i=1:size(dataTestFreqTrain1,3)
    X1= dataTestFreqTrain1(:,:,i);
    X2= dataTestFreqTrain2(:,:,i);
    for j=1:size(X1,2)
        tp1Test1(:,j)= myfeatureExtraction(X1(:,j));
        tp1Test2(:,j)= myfeatureExtraction(X2(:,j));
    end
    Features1FreqTest(:,i)= [tp1Test1(:); tp1Test2(:)];
end
dataTestFreq = Features1FreqTest;
%% Features Selection By PCA
data1 = [data1Time ; data1Freq];
data2 = [data2Time ; data2Freq];
data3 = [data3Time ; data3Freq];
DataTest = [dataTestTime ; dataTestFreq];
data = [data1 data2 data3 DataTest];
% data = [data1 data2 data3 ; ones(1,8) 2*ones(1,8) 3*ones(1,8)];
% d = 50; % #Dimension or #Features
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

% save allfeatures data1 data2 data3
cols = size(data1,2);
P = randperm(cols);
data1N = data1(:,P);
data2N = data2(:,P);
data3N = data3(:,P);
data1 = [data1 , data1N];
data2 = [data2 , data2N];
data3 = [data3 , data3N];


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
    mdl1= fitcknn(traindata',trainlabel,'NumNeighbors',3);
    mdl2= fitcknn(traindata',trainlabel,'NumNeighbors',5);
    mdl3= fitctree(traindata',trainlabel);
    kernel='linear';
    mdl4 = mymultisvmtrainOVO(traindata,trainlabel,kernel);
    mdl5= fitcnb(traindata',trainlabel);
    mdl6= fitcdiscr(traindata',trainlabel);
    %% step 3: test trained model using test data
    R(1,:)= predict(mdl1,testdata');
    RTest(1,:)= predict(mdl1,DataTest');
    R(2,:)= predict(mdl2,testdata');
    RTest(2,:)= predict(mdl2,DataTest');
    R(3,:)= predict(mdl3,testdata');
    RTest(3,:)= predict(mdl3,DataTest');
    R(4,:)= mymultisvmclassifyOVO(mdl4,testdata);
    RTest(4,:)= mymultisvmclassifyOVO(mdl4,DataTest);
    R(5,:)= predict(mdl5,testdata');
    RTest(5,:)= predict(mdl5,DataTest');
    R(6,:)= predict(mdl6,testdata');
    RTest(6,:)= predict(mdl6,DataTest');
%     R(5,:)= predict(mdl7,testdata')';
%     RTest(5,:)= predict(mdl7,DataTest');
%     R(7,:)= mySLPtest(mdl7,testdata);
%     RTest(7,:)= mySLPtest(mdl7,DataTest);
    %%
    userlabel=unique(trainlabel);
    output = myMejorityVoting(R,userlabel);
    outputTest(C1,:) = myMejorityVoting(RTest,userlabel);
    C1 = C1 + 1;
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
load ResultTest3
ResultTest3(15,:) = Total;
save ResultTest3 ResultTest3