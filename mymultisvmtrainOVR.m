function [mdl] = mymultisvmtrainOVR(Xtrain,Ytrain,kernel)
userlabel= unique(Ytrain);
%% 1 vs (2,3)
data1= Xtrain(:,Ytrain==userlabel(1));
data2= Xtrain(:,Ytrain~=userlabel(1));
traindata=[data1,data2];
trainlabel= [ones(1,size(data1,2)),2*ones(1,size(data2,2))];

mdl.svm1 = fitcsvm(traindata',trainlabel,'KernelFunction',kernel,'Standardize',1);
%% 2 vs (1,3)
data1= Xtrain(:,Ytrain==userlabel(2));
data2= Xtrain(:,Ytrain~=userlabel(2));
traindata=[data1,data2];
trainlabel= [ones(1,size(data1,2)),2*ones(1,size(data2,2))];

mdl.svm2 = fitcsvm(traindata',trainlabel,'KernelFunction',kernel,'Standardize',1);
%% 3 vs (1,2)
data1= Xtrain(:,Ytrain==userlabel(3));
data2= Xtrain(:,Ytrain~=userlabel(3));
traindata=[data1,data2];
trainlabel= [ones(1,size(data1,2)),2*ones(1,size(data2,2))];

mdl.svm3 = fitcsvm(traindata',trainlabel,'KernelFunction',kernel,'Standardize',1);
mdl.userlabel=userlabel;
end

