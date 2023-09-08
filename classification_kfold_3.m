clc
clear
close all
%% www.onlinebme.com
load('allfeatures.mat')
clearvars -except data1 data2 data3
%% load selected features address
% load indx_selection_KNN3
load indx_selection_svm3
%% step 1: devide data into train and test data
% k-fold cross validation(k=5)
k=5;
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
    
    %% feature selection using sffs
    numf=20;
    traindata= traindata(sel(1:numf),:);
    testdata =testdata(sel(1:numf),:);
    %% step 2: train model using train data and train label
    %     mdl= fitcknn(traindata',trainlabel,'NumNeighbors',5);
    mdl = mymultisvmtrainOVO(traindata,trainlabel,'linear');
    %% step 3: test trained model using test data
    %     output= predict(mdl,testdata')';
    output = mymultisvmclassifyOVO(mdl,testdata);
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


