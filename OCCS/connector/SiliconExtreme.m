classdef SiliconExtreme<handle
   
    properties
        port
        on_off
    end
    
    methods
        function obj = SiliconExtreme(COM_order)
            obj.port = serial(sprintf('COM%d',COM_order), 'baudrate', 115200,...
                'parity','none', 'databits', 8, 'stopbits',1);
            obj.on_off=0;
            obj.open();
        end

        function IsSuccess = open(obj)
            if obj.on_off==1
                IsSuccess=1;
                return
            end
            fopen(obj.port);
            obj.on_off=1;
            IsSuccess=1;
        end

        function IsSuccess = close(obj)
            if obj.on_off==0
                IsSuccess=1;
                return
            end
            fclose(obj.port);
            obj.on_off=0;
            IsSuccess=1;
        end
        
        function answer = query(obj,text)
            fprintf(obj.port,text);
            answer=fscanf(obj.port);
        end

        function V = getV(obj,channel)
            V=obj.query(sprintf('V%d?',channel));
        end

        function IsSuccess = setV(obj,channel,V)
            answer=obj.query(sprintf('V%d=%.4f',channel,V));
            IsSuccess=strcmp(answer,sprintf('OK\n'));
        end

        function I = getI(obj,channel)
            I=obj.query(sprintf('I%d?',channel));
        end

        function IsSuccess = setI(obj,channel,I)
            answer=obj.query(sprintf('I%d=%.4f',channel,I));
            IsSuccess=strcmp(answer,sprintf('OK\n'));
        end

        function P = getP(obj,channel)
            P=obj.query(sprintf('P%d?',channel));
        end

        function IsSuccess = setVmax(obj,channel,Vmax)
            answer=obj.query(sprintf('VMAX%d=%.4f',channel,Vmax));
            IsSuccess=strcmp(answer,sprintf('OK\n'));
        end

        function IsSuccess = setImax(obj,channel,Imax)
            answer=obj.query(sprintf('IMAX%d=%.4f',channel,Imax));
            IsSuccess=strcmp(answer,sprintf('OK\n'));
        end

        function IsSuccess = setVmax_all(obj,Vmax)
            answer=obj.query(sprintf('VMAXALL=%.4f',Vmax));
            IsSuccess=strcmp(answer,sprintf('OK\n'));
        end

        function IsSuccess = setImax_all(obj,Imax)
            answer=obj.query(sprintf('IMAXALL=%.4f',Imax));
            IsSuccess=strcmp(answer,sprintf('OK\n'));
        end
    end
end

