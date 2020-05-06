%本程序需根据需要改class和nf两处即可。classX为txt文件名，nf为文件内蛋白质条数
clear all;
clc;
class=7;
nf=20;% the number of the files
fixpart1='class';
fixpart2='.txt';
fname=sprintf('%s%d%s',fixpart1,class,fixpart2);%将class1.txt文件内容format成一维数组
ex=importdata(fname);
line_number=size(ex,1);%txt文件的行数。
protein_index=zeros(1,line_number);%初始化蛋白质标识为1*line_number零向量
index=zeros(1,nf);
j=1;
for i=1:line_number
    if(size(strfind(ex {i},'>'))>0)
    protein_index(i)=strfind(ex{i},'>');%存下蛋白质标识所在行号
    end
    if(protein_index(i)~=0)
        index(j)=i;%存下该条蛋白质标识所在行号
        j=j+1;
    end
end
indicator={'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'};
frequent_matrix=zeros(nf,400);
%计算蛋白质出现的频率，每一条蛋白质存储在frequent_matrix的一行。
for j=1:nf
frequent_number=zeros(1,400);
number=j;
str='';
   if(number<nf)
      for i=index(number)+1:index(number+1)-1  %第number条蛋白质所跨越的所有行
          str=[str,ex{i}];   %把第number条蛋白质序列存在字符串str里
      end
   end
   
   if (number==nf) %处理最后一条蛋白质
      for i=index(nf)+1:size(ex,1)
          str=[str,ex{i}]; 
      end
   end
  
   h=1;
   for n=1:20
       for k=1:20
           Dipstr=strcat(indicator{n},indicator{k});  %定义二联体；注意是{}，否则会出错
           frequent_number(h)=size(strfind(str,Dipstr),2)/(size(str,2)-1); %size(str,2)-1表示该条蛋白质二联体的长度
           h=h+1;
       end
   end

frequent_matrix(j,:)=frequent_number;
end

%outpart1='Protein_';
%outpart2='DipC.txt';
%outputname=sprintf('%s%s',outpart1,outpart2);
%dlmwrite(outputname,frequent_matrix);
% frequent_matrix(nf,:)=frequent_number;
%y=frequent_matrix(:,2:401);
%plot(1:20,y);




                            