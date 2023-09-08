clc
clear
close all
%% Load Data
load('sub13.mat')
% plot(DataTrain_Hb(:,15,7))
%% step 1-0: channel selection
load('indxChn_anova_g.mat')
numChn= 10;
%% step 1:  cut Labels from DataTrain
Block_Train= ClassLabel_Train;
c1=0;
c2=0;
c3=0;
for i=1:numel(Block_Train)
    if Block_Train(i) ==1
        c1=c1+1;
        data1(:,:,c1)=DataTrain_Hb(:,sel(1:numChn),i);
    elseif Block_Train(i)== 2
        c2=c2+1;
        data2(:,:,c2)=DataTrain_Hb(:,sel(1:numChn),i);
    elseif Block_Train(i)== 3
        c3=c3+1;
        data3(:,:,c3)=DataTrain_Hb(:,sel(1:numChn),i);
    end
end
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
    data1Freq(:,:,c1) = fx_a;
    data2Freq(:,:,c1) = fx_b;
    data3Freq(:,:,c1) = fx_c;
    c1 = c1 + 1;
end

for i=1:size(data1Freq,3)
    X1= data1Freq(:,:,i);
    X2= data2Freq(:,:,i);
    X3= data3Freq(:,:,i);
    for j=1:size(X1,2)
        tp1(:,j)= myfeatureExtraction(X1(:,j));
        tp2(:,j)= myfeatureExtraction(X2(:,j));
        tp3(:,j)= myfeatureExtraction(X3(:,j));
    end
    Features1(:,i)= tp1(:);
    Features2(:,i)= tp2(:);
    Features3(:,i)= tp3(:);
end
data1Freq = Features1;
data2Freq = Features2;
data3Freq = Features3;
%% Features Selection By PCA
data1 = [data1Time ; data1Freq];
data2 = [data2Time ; data2Freq];
data3 = [data3Time ; data3Freq];
data = [data1 data2 data3];
d = 25; % #Dimension or #Features
w = myPCA(data, d);
f1 = w' * data;

data1 = f1(:,1:8);
data2 = f1(:,9:16);
data3 = f1(:,17:24);

% data1 = repmat(data1,1,2);
% data2 = repmat(data2,1,2);
% data3 = repmat(data3,1,2);

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
    
    %%
    div=0.5;
    num= round(div* size(traindata,2));
    indx= randperm(size(traindata,2));
    datavalid= traindata(:,indx(num+1:end));
    dvalid= trainlabel(indx(num+1:end));
    
    datatrain= traindata(:,indx(1:num));
    dtrain=  trainlabel(indx(1:num));
    %% step 2: train calssifier using train data & train label
    % stacking : first level training
    mdl1= fitcknn(datatrain',dtrain,'NumNeighbors',3);
    mdl2= fitctree(datatrain',dtrain);
%     mdl3= fitcdiscr(datatrain',dtrain);
    mdl4= fitcnb(datatrain',dtrain);
    kernel='linear';
    mdl5 = mymultisvmtrainOVO(datatrain,dtrain,kernel);
    
    
    %% second level training
    Xtrain(1,:)= predict(mdl1,datavalid');
    Xtrain(2,:)= predict(mdl2,datavalid');
%     Xtrain(3,:)= predict(mdl3,datavalid');
    Xtrain(3,:)= predict(mdl4,datavalid');
    Xtrain(4,:)= mymultisvmclassifyOVO(mdl5,datavalid);
    
    mdl = mySLPtrainR(Xtrain,dvalid);
    
    %% step 3:test trained classifier using test data
    R(1,:)= predict(mdl1,testdata');
    R(2,:)= predict(mdl2,testdata');
%     P(3,:)= predict(mdl3,datatest');
    R(3,:)= predict(mdl4,testdata');
    R(4,:)= mymultisvmclassifyOVO(mdl5,testdata);
    %% weighted voting : combine votes
    output = mySLPtestR(mdl,R);
    output = round(output);
    C= confusionmat(testlabel,output);
    Totalaccuracy(i)=  sum(diag(C)) / sum(C(:)) *100;
    accuracy1(i)= C(1,1) / sum(C(1,:)) *100;
    accuracy2(i)= C(2,2) / sum(C(2,:)) *100;
    accuracy3(i)= C(3,3) / sum(C(3,:)) *100;
    Ct= Ct+C;
end
Ct
disp(['total Accuracy: ',num2str(mean(Totalaccuracy)) ,'%'])
disp(['accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['accuracy3: ',num2str(mean(accuracy3)) ,'%'])



