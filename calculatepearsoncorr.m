function [linearcorr bestbeta bestquality] = calculatepearsoncorr(quality,humanscores,drawing)

x = quality; 
y = humanscores; 
bestbeta = zeros(5,1);
bestpcc = 0;
bestquality = quality;
itr =1;
for fact = 0.01:0.1:10
    itr = itr+1;
beta0 = [max(y) min(y) fact*mean(x) 1 0];
%[msg, msgid] = lastwarn
%s = warning('off','stats:nlinfit:IllConditionedJacobian');
warning off all
b_blur = nlinfit(x,y,@logistic_fun,beta0);
x_hat = logistic_fun(b_blur,x);
Correlation(itr) = corr(x_hat,y);
if Correlation(itr) > bestpcc
    bestpcc = Correlation(itr);
    bestbeta = b_blur;
    bestquality = x_hat;
end
% [sorted_quality ind] = sort(quality);
% x_hat_sorted = x_hat(ind);
% scatter(sorted_quality,x_hat_sorted)
% hold on
% scatter(quality,humanscores,'r')
% figure(1)
end

linearcorr = max(Correlation);

if drawing
    figure;hold on;
    xline = [min(quality):0.01:max(quality)];
    yline = logistic_fun(bestbeta,xline);
    plot(xline,yline);
    plot(quality,humanscores,'*k');
    title(['PCC=' num2str(linearcorr)]);
end

