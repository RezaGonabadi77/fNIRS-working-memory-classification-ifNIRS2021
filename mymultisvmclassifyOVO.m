function [output] = mymultisvmclassifyOVO(mdl,Xtest)
userlabel= mdl.userlabel;
%%
svm1=mdl.svm1;
output1 = predict(svm1,Xtest')';

svm2=mdl.svm2;
output2 = predict(svm2,Xtest')';

svm3=mdl.svm3;
output3 = predict(svm3,Xtest')';
%%
temp=[output1;output2;output3];
for i= 1:numel(userlabel)
    num(i,:)= sum( temp== userlabel(i));
end
[mx,ind]= max(num);
output= zeros(1,size(Xtest,2));

for i= 1:numel(userlabel)
    index = find(ind==i);
   output(index)= userlabel(i);
end

end








