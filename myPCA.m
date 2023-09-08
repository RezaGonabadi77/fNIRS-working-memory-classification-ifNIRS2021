function W=myPCA(Xtrain,m)
% pca function
% Xtrain is a d*N matrix
% m is the number of igenvectors
% step 1: means  zeros
meanX=mean(Xtrain');
Xtrain=Xtrain-repmat(meanX',1,size(Xtrain,2));
% plot(Xtrain(1,:),Xtrain(2,:),'r.','DisplayName','Xtrain')
% step 2: calculatin covariance matrix
C=cov(Xtrain');
% step 3: diagnalization
[U,V]=eig(C);
V=diag(V);
% step 4: sorting data
[V,ind]=sort(V,'descend');
U=U(:,ind);
W=U(:,1:m);
end

