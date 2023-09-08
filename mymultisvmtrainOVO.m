function [mdl] = mymultisvmtrainOVO(Xtrain,Ytrain,kernel)
userlabel= unique(Ytrain);
y1=userlabel(1);
y2=userlabel(2);
y3=userlabel(3);
%% 1 vs (2)
data1= Xtrain(:,Ytrain==userlabel(1));
data2= Xtrain(:,Ytrain==userlabel(2));
traindata=[data1,data2];
trainlabel= [y1*ones(1,size(data1,2)),y2*ones(1,size(data2,2))];

mdl.svm1 = fitcsvm(traindata',trainlabel,'KernelFunction',kernel,'Standardize',1);
%% 1 vs (3)
data1= Xtrain(:,Ytrain==userlabel(1));
data2= Xtrain(:,Ytrain==userlabel(3));
traindata=[data1,data2];
trainlabel= [y1*ones(1,size(data1,2)),y3*ones(1,size(data2,2))];

mdl.svm2 = fitcsvm(traindata',trainlabel,'KernelFunction',kernel,'Standardize',1);
%% 2 vs (3)
data1= Xtrain(:,Ytrain==userlabel(2));
data2= Xtrain(:,Ytrain==userlabel(3));
traindata=[data1,data2];
trainlabel= [y2*ones(1,size(data1,2)),y3*ones(1,size(data2,2))];

mdl.svm3 = fitcsvm(traindata',trainlabel,'KernelFunction',kernel,'Standardize',1);
mdl.userlabel=userlabel;

end

