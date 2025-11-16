%TEST2 M-Code for communicating with an instrument.
%
%   This is the machine generated representation of an instrument control
%   session. The instrument control session comprises all the steps you are
%   likely to take when communicating with your instrument. These steps are:
%   
%       1. Create an instrument object
%       2. Connect to the instrument
%       3. Configure properties
%       4. Write and read data
%       5. Disconnect from the instrument
% 
%   To run the instrument control session, type the name of the M-file,
%   Test2, at the MATLAB command prompt.
% 
%   The M-file, TEST2.M must be on your MATLAB PATH. For additional information 
%   on setting your MATLAB PATH, type 'help addpath' at the MATLAB command 
%   prompt.
% 
%   Example:
%       test2;
% 
%   See also SERIAL, GPIB, TCPIP, UDP, VISA.
% 
 
%   Creation time: 06-May-2014 08:50:29

% Find a GPIB object.

% main settings
%-------------------------------------------------------------------------------
format long

lamstar = 1548;
lamstop = 1552; % optical source

% res= 0.5;
res =0.02;

%sensitivity = 'normal';
sensitivity = 'high2';
% sensitivity = 'normal';
num_inset=5;
pointnum = (lamstop-lamstar)*num_inset/res+1;
speed = '2x';

reflevel_up = -0;
reflevel_down = -100;
channel = 'a';                                                                                                                                                                                                                                                                          
r = 1;  % 1:制定波长范围；0：默认光源范围
record = 1; %是否保存数据
normalization = 0; %是否归一化
filename = ['D:\测试\实验测试数据\6.10\amzi2','_',  num2str(lamstar),'_',num2str(lamstop),'_',num2str(res),'_',sensitivity,'.txt'];
% refpath = ['F:\Users\Zhangchangping\20220601\wss_test1','_',num2str(lamstar),'_',num2str(lamstop),'_',num2str(res),'_',sensitivity,'.txt'];

% Create the GPIB object if it does not exist
% otherwise use the object that was found.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
%---------------------------------------------------------------------------------
BoardIndex=0; PrimaryAddress=1;
% BoardIndex=00; PrimaryAddress=1;
% BoardIndex=3; PrimaryAddress=01;
% BoardIndex=02; PrimaryAddress=00;
obj1 = instrfind('Type', 'gpib', 'BoardIndex', BoardIndex, 'PrimaryAddress', PrimaryAddress, 'Tag', '');
if isempty(obj1)
    obj1 = gpib('NI', BoardIndex, PrimaryAddress);
else
    fclose(obj1);
    obj1 = obj1(1)
    a=1
end
set(obj1, 'InputBufferSize', 180009*3);
set(obj1, 'OutputBufferSize', 180009*3);
set(obj1, 'Timeout', 150);
% Communicating with instrument object, obj1.
%-----------------------------------------------------------------
fopen(obj1);
fprintf(obj1, [':sens:wav:star ',num2str(lamstar),'nm']);
fprintf(obj1, [':sens:wav:stop ',num2str(lamstop),'nm']);
fprintf(obj1, [':disp:trace:y1:rlev ',num2str(reflevel_up),'dbm']);
fprintf(obj1,[':sens:swe:points ',num2str(pointnum)]);
% fprintf(obj1,':sens:swe:points:auto on');
fprintf(obj1,[':sens:band:res ',num2str(res),'nm']);
fprintf(obj1,[':sens:sens ',sensitivity]);
fprintf(obj1,[':trac:attr:tr',channel,' write']);
fprintf(obj1, ':initiate:smode single');
fprintf(obj1,[':sens:swe:spe ',speed]);
fprintf(obj1,[':trac:stat:tr',channel,' on']);
fprintf(obj1,'*CLS');
fprintf(obj1,':init');
fprintf(obj1,':stat:oper:even?');
finish = fscanf(obj1);
while ~finish
    fprintf(obj1,':stat:oper:even?')
    fscanf(obj1);
end
fprintf(obj1,':format:data ascii');
fprintf(obj1,[':trac:y? tr',channel]);
Power = fscanf(obj1);
fprintf(obj1,[':trac:x? tr',channel]);
Lambda = fscanf(obj1);

%% Preset the parameter for alignment
%-------------------------------------------------------------
if r==1
    fprintf(obj1, [':sens:wav:star ',num2str(lamstar),'nm']);
    fprintf(obj1, [':sens:wav:stop ',num2str(lamstop),'nm']);
elseif r==0
    fprintf(obj1, [':sens:wav:star ','1520','nm']);
    fprintf(obj1, [':sens:wav:stop ','1610','nm']);
end
fprintf(obj1, ':disp:trace:y1:rlev -10dbm'); %%% 这里是OSA上面显示的ref
fprintf(obj1,':sens:band:res 0.1nm');
fprintf(obj1,':sens:sens normal');
fprintf(obj1,':sens:swe:points:auto on');
fprintf(obj1,':sens:swe:spe 2x');
fprintf(obj1, ':initiate:smode repeat');
fprintf(obj1, ':init');
pause(0.5);
fclose(obj1);
delete(obj1);

% Dealing with the data
%----------------------------------------------------------
Points = pointnum;
Power = char(Power);
Lambda = char(Lambda);
P = zeros(1,Points);
L = zeros(1,Points);
I = strfind(Power,',');
P(1) = str2double(Power(1:I(1)-1));
for Loop = 1:length(I)-1
    P(Loop+1) = str2double(Power(I(Loop)+1:I(Loop+1)-1));
end
P(end) = str2double(Power(I(end)+1:end));
I = strfind(Lambda,',');
L(1) = str2double(Lambda(1:I(1)-1));
for Loop = 1:length(I)-1
    L(Loop+1) = str2double(Lambda(I(Loop)+1:I(Loop+1)-1));
end
L(end) = str2double(Lambda(I(end)+1:end));
L=L*1e9;
data=[L',P'];

% Save the data
%------------------------------------------------------------
if record   
    save(filename,'-ascii','data');
end

%% Plot the data
%------------------------------------------------------------
figure(1);
set(gcf,'unit','centimeters','position',[5,5,15,10]);
hold on;
if normalization
%     refte=load('E:\Sitao Chen\Spectroscope\20150204_PbsPrAwg_Final\TE_str1_1530-1560_T1_0.5nm.txt');
%     reftm=load('E:\Sitao Chen\Spectroscope\20150204_PbsPrAwg_Final\FPTM_str1_1530-1560_T1_0.5nm.txt');
%     refl1=refte(:,1);
%     refp1=refte(:,2);
%     refl2=reftm(:,1);
%     refp2=reftm(:,2);
%     data=[(refl1+refl2)/2,(refp1+refp2)/2];
%     save('E:\Sitao Chen\Spectroscope\20150204_PbsPrAwg_Final\TETM_str1_1530-1560_0.5nm.txt','-ascii','data');
    
    ref=load(refpath);
    refy=ref(:,2);
    refy=refy';
%     plot(L,P-refy,'k');
    plot(L,P-refy,'linewidth',2);
    grid on;
    axis ([lamstar lamstop reflevel_down reflevel_up]);
else
    plot(L,P,'linewidth',2);
    grid on;
    axis ([lamstar lamstop -100 reflevel_up]);
end
%%