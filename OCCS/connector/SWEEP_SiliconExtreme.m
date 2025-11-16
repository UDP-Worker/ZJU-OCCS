
COM_order = 5; % 使用前用上机位程序或keysight确定串口号，newsetup可能是16
obj1 = instrfind('Port','COM5');
if isempty(obj1)
    obj1 = SiliconExtreme(COM_order);
else
    fclose(obj1);
    obj1 = SiliconExtreme(COM_order);
end


open(obj1);
Imax=12; % mA,可以设置的最大，不一定可以达到，取决于最大输出功率
Vmax=6;  % V，可以设置的最大，不一定可以达到，取决于最大输出功率
setImax(obj1,31,Imax);
setVmax(obj1,31,Vmax);
setImax(obj1,30,Imax);
setVmax(obj1,30,Vmax);
setImax(obj1,29,Imax);
setVmax(obj1,29,Vmax);
setImax(obj1,28,Imax);
setVmax(obj1,28,Vmax);
setImax(obj1,27,Imax);
setVmax(obj1,27,Vmax);
setImax(obj1,26,Imax);
setVmax(obj1,26,Vmax);


% 1-MRR (channel31 0.9V 0.85V 0.75V)
    % setV(obj1,31,0.85);
    % setV(obj1,30,0.95);
    % setV(obj1,29,1.5);
    % setV(obj1,28,1.5);
    % setV(obj1,27,1.5);
    % setV(obj1,26,1.5);


% 2-MRR (channel31 0.67V 29 0.85V 26 0.78V持平深
%        channel31 0.67V 29 0.85V 26 0.85V持平浅
%        channel31 0.67V 29 0.85V 26 0.90V方波)
    % setV(obj1,31,0.67);
    % setV(obj1,30,1.51);
    % setV(obj1,29,0.85);
    % setV(obj1,28,1.51);
    % setV(obj1,27,1.51);
    % setV(obj1,26,0.90);
    % setV(obj1,25,1.51);
    % setV(obj1,24,1.51);
    % setV(obj1,23,1.51);
    % setV(obj1,22,1.51);
    % setV(obj1,21,1.51);

% a1-mzi (channel29 1.01 0.95 0.85)
      % setV(obj1,31,1.51);
      % setV(obj1,30,0.35);
      % setV(obj1,29,1.01);
      % setV(obj1,28,1.51);
      % setV(obj1,27,1.51);
      % setV(obj1,26,1.51);
      % setV(obj1,25,0.35);
      % setV(obj1,24,1.1);

% 3-tap
    % setV(obj1,31,1.43);
    % setV(obj1,30,0.44);
    % setV(obj1,29,1.1);
    % setV(obj1,28,0.46);
    % setV(obj1,27,1.1);
    % setV(obj1,26,1.53);
    % setV(obj1,25,1.48);
    % setV(obj1,24,1.47);
    % setV(obj1,23,1.45);
    % setV(obj1,22,0.48);
    % setV(obj1,21,0.95);
    % setV(obj1,20,0.4);
    % setV(obj1,19,1.1);

    % setV(obj1,31,0);
    % setV(obj1,30,0);
    % setV(obj1,29,1.51);
    % setV(obj1,28,0.4);
    % setV(obj1,27,1.0);
    % setV(obj1,26,0);
    % setV(obj1,25,1.51);
    % setV(obj1,24,1.51);
    % setV(obj1,23,0);
    % setV(obj1,22,0);
    % setV(obj1,21,1.51);
    % setV(obj1,20,0.4);
    % setV(obj1,19,1.02);

% a2-mzi(channel19 1.02(频率测量) 1.05 1.1 1.15)
    % setV(obj1,31,1.51);
    % setV(obj1,30,0.4);
    % setV(obj1,29,0.4);
    % setV(obj1,28,0.4);
    % setV(obj1,27,1.0);
    % setV(obj1,26,1.51);
    % setV(obj1,25,0);
    % setV(obj1,24,1.51);
    % setV(obj1,23,1.51);
    % setV(obj1,22,0.4);
    % setV(obj1,21,0.4);
    % setV(obj1,20,0.4);
    % setV(obj1,19,1.02);

% ramzi(channel26 0.9 1.0 1.2)
    % setV(obj1,31,1.51);
    % setV(obj1,30,0.4);
    % setV(obj1,29,1.1);
    % setV(obj1,28,0.4);
    % setV(obj1,27,0.4);
    % setV(obj1,26,1.1);
    % setV(obj1,25,0.4);
    % setV(obj1,24,0.4);
    % setV(obj1,23,1.51);
    % setV(obj1,22,1.51);
    % setV(obj1,21,1.51);
    % setV(obj1,20,1.51);
    % setV(obj1,19,1.51);

% 归一化
    setV(obj1,31,0);
    setV(obj1,30,0);
    setV(obj1,29,0);
    setV(obj1,28,0);
    setV(obj1,27,0);
    setV(obj1,26,0);
    setV(obj1,25,0);
    setV(obj1,24,0);
    setV(obj1,23,0);
    setV(obj1,22,0);
    setV(obj1,21,0);
    setV(obj1,20,0);
    setV(obj1,19,0);

       pause(0.5); %只是怕不稳，设置了每次读数的间隔0.5s
        % Vt = getV(obj1,channel);
        % It = getI(obj1,channel);

    


close(obj1);
