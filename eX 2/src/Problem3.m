clc;clear all

data = xlsread('got_data1.xls');

seasons = data(:,7);
episodes = data(:,2);
screentime = data(:,3);
family = data(:,5);
screenTimeMale = sum(screentime(data(:,6)==1));
screenTimeFemale = sum(screentime(data(:,6)==2));

sexCount = [sum(data(:,6)==1),sum(data(:,6)==2)];
screenTimeSex = [screenTimeMale,screenTimeFemale];

screenTimeLannister = sum(screentime(data(:,5)==1));
screenTimeStark = sum(screentime(data(:,5)==2));
screenTimeOther = sum(screentime(data(:,5)==3));

screenTimeFamily = [screenTimeLannister,screenTimeStark,screenTimeOther];

screenTimeSolo = data(:,3);
%% Draw histogram of number of family members(1- Lannister,2-Stark,3-Other)
hist(data(:,5))

plot(episodes,screentime,'o')

bar(screenTimeSex,0.1)

bar(screenTimeFamily,0.1)

bar(screenTimeSolo,0.4)
