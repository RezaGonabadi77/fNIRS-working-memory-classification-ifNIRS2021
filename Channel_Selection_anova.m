clc
clear
close all
%% setup
load('sub15_Test_3_Competition_2021.mat')
%% step 1:  cut Labels from DataTrain
Block_Train= ClassLabel_Train;
c1=0;
c2=0;
c3=0;
for i=1:numel(Block_Train)
    if Block_Train(i) ==1
        c1=c1+1;
        data1(:,:,c1)=DataTrain_HbO2(70:470,:,i);
    elseif Block_Train(i)== 2
        c2=c2+1;
        data2(:,:,c2)=DataTrain_HbO2(70:470,:,i);
    elseif Block_Train(i)== 3
        c3=c3+1;
        data3(:,:,c3)=DataTrain_HbO2(70:470,:,i);
    end
end

%% step 2: feature extaction
for i=1:size(data2,3)
    X1=data1(:,:,i);
    X2=data2(:,:,i);
    X3=data3(:,:,i);
    for j=1:size(X1,2)
        tp1(:,j)= wentropy(X1(:,j),'log energy');
        tp2(:,j)= wentropy(X2(:,j),'log energy');
        tp3(:,j)= wentropy(X3(:,j),'log energy');
    end

    Features1(:,i)=tp1(:);
    Features2(:,i)=tp2(:);
    Features3(:,i)=tp3(:);
end
%% channel selection using Anova
dataset1=Features1;
dataset2=Features2;
dataset3=Features3;
Totaldata=[dataset1,dataset2,dataset3];
label=[ ones(1,size(dataset1,2)),2*ones(1,size(dataset2,2)),3*ones(1,size(dataset2,2))];
% claculate each channels pvalue
for j=1:size(Totaldata,1)
    xj= Totaldata(j,:);
    P(j)= anova1(xj,label,'off');
end
%% channel sorting based on p-value
[P,ind]= sort(P,'ascend');
sel= ind;
save indxChn_anova_g sel