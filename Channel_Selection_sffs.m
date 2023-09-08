clc
clear
close all
%% Load Data
load('sub06.mat')

%% step 1:  cut Labels from DataTrain
Block_Train= ClassLabel_Train;
c1=0;
c2=0;
c3=0;
for i=1:numel(Block_Train)
    if Block_Train(i) ==1
        c1=c1+1;
        data1(:,:,c1)=DataTrain_HbO2(:,:,i);
    elseif Block_Train(i)== 2
        c2=c2+1;
        data2(:,:,c2)=DataTrain_HbO2(:,:,i);
    elseif Block_Train(i)== 3
        c3=c3+1;
        data3(:,:,c3)=DataTrain_HbO2(:,:,i);
    end
end

%% step 4: channel selection
indx_chn= 1:size(data3,2);
sel=[];
max_numf=10;
for iter=1:max_numf
    c=0;
    for Chnum=indx_chn
        c=c+1;
        indx_cond=[sel,Chnum];
        %% step 3: feature extaction
        for i=1:size(data2,3)
            X1=data1(:,:,i);
            X2=data2(:,:,i);
            X3=data3(:,:,i);
            count=0;
            for j=indx_cond
                count= count+1;
                tp1(:,count)= myfeatureExtraction(X1(:,j));
                tp2(:,count)= myfeatureExtraction(X2(:,j));
                tp3(:,count)= myfeatureExtraction(X3(:,j));
            end
            
            Features1(:,i)=tp1(:);
            Features2(:,i)=tp2(:);
            Features3(:,i)=tp3(:);
        end
        %% channel selection using sffs
        data1=Features1;
        data2=Features2;
        data3=Features3;
        %% step 1: devide data into train(70%) and test(30%)
        div=0.7;
        num= round(div* size(data1,2));
        
        datatrain1= data1(:,(1:num));
        datatest1 = data1(:,(num+1:end));
        
        datatrain2= data2(:,(1:num));
        datatest2 = data2(:,(num+1:end));
        
        datatrain3= data3(:,(1:num));
        datatest3 = data3(:,(num+1:end));
        
        
        datatrain=[datatrain1,datatrain2,datatrain3];
        dtrain=[ ones(1,size(datatrain1,2)),2*ones(1,size(datatrain2,2)),3*ones(1,size(datatrain3,2))];
        
        
        datatest=[datatest1,datatest2,datatest3];
        dtest=[ ones(1,size(datatest1,2)),2*ones(1,size(datatest2,2)),3*ones(1,size(datatest3,2))];
        %% step 2: train classifier using train data & train label
%         mdl=fitcsvm(datatrain',dtrain,'Standardize',1);
        kernel='linear';
        mdl = mymultisvmtrainOVO(datatrain,dtrain,kernel);
        %% step 3: test trained model using test data
%         output =predict(mdl,datatest');
        output=mymultisvmclassifyOVO(mdl,dtest);
        %% step 4: validation
        Cmx= confusionmat(dtest,output);
        % total accuracy
        accuracy= sum(diag(Cmx)) / sum(Cmx(:)) *100;
        perfomance(c)= accuracy;
    end
    [perfomance,ind]= sort(perfomance,'descend');
    bestperfomance(iter)= perfomance(1);
    sel=[sel,indx_chn(ind(1))];
    indx_chn(ind(1))=[];
    perfomance=[];
    %% ploting
    subplot(1,2,1)
    plot(positions(1,sel),positions(2,sel),'ob','linewidth',2,...
        'MarkerSize',10,'markerfacecolor','g')
    hold on
    grid on
    grid minor
    subplot(1,2,2)
    plot(bestperfomance(1:iter),'b','linewidth',2)
    hold on
    plot(bestperfomance(1:iter),'ro','linewidth',2,'markersize',10)
    %     text(iter,bestperfomance(iter)+(randn(1)*0.3),num2str(sel),'fontsize',15)
    grid on
    grid minor
    drawnow
    Features1 = [];
    Features2 = [];
    tp1 = [];
    tp2 = [];
end
save indx_selection_sffs_svm_g sel bestperfomance





