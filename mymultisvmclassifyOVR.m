function [output] = mymultisvmclassifyOVR(mdl,Xtest)
userlabel= mdl.userlabel;
%%
svm1=mdl.svm1;
output1 = predict(svm1,Xtest')';

svm2=mdl.svm2;
output2 = predict(svm2,Xtest')';

svm3=mdl.svm3;
output3 = predict(svm3,Xtest')';

for i=1:size(Xtest,2)
    y1= output1(i);
    y2= output2(i);
    y3= output3(i);
    if y1==1
        output(i) = userlabel(1);
    elseif y2==1
        output(i) = userlabel(2);
    elseif y3==1
        output(i) = userlabel(3);
    else
        output(i) = nan;
%         output(i) = userlabel(3);
    end
end
end
