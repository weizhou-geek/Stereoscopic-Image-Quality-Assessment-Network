%compute correlation

clear all
close all


%% ALL
load('predictScore_testLIVE1_model_200.mat')
load('MOS_test.mat')
cnt1=1;
avg_score=ones(73,1);
for cnt=1:220:length(score)
    avg_score(cnt1)=mean(score(cnt:cnt+219));
    cnt1=cnt1+1;
end

cnt2=1;
avg_MOS=ones(73,1);
for cnt=1:220:length(MOS)
    avg_MOS(cnt2)=mean(MOS(cnt:cnt+219));
    cnt2=cnt2+1;
end

avgScore=avg_score;
avgMOS=avg_MOS;

SROCC_all = corr(avgScore, avgMOS, 'type', 'Spearman')
PCC_all = calculatepearsoncorr(avgScore, avgMOS, 0)


