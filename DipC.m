%�������������Ҫ��class��nf�������ɡ�classXΪtxt�ļ�����nfΪ�ļ��ڵ���������
clear all;
clc;
class=7;
nf=20;% the number of the files
fixpart1='class';
fixpart2='.txt';
fname=sprintf('%s%d%s',fixpart1,class,fixpart2);%��class1.txt�ļ�����format��һά����
ex=importdata(fname);
line_number=size(ex,1);%txt�ļ���������
protein_index=zeros(1,line_number);%��ʼ�������ʱ�ʶΪ1*line_number������
index=zeros(1,nf);
j=1;
for i=1:line_number
    if(size(strfind(ex {i},'>'))>0)
    protein_index(i)=strfind(ex{i},'>');%���µ����ʱ�ʶ�����к�
    end
    if(protein_index(i)~=0)
        index(j)=i;%���¸��������ʱ�ʶ�����к�
        j=j+1;
    end
end
indicator={'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'};
frequent_matrix=zeros(nf,400);
%���㵰���ʳ��ֵ�Ƶ�ʣ�ÿһ�������ʴ洢��frequent_matrix��һ�С�
for j=1:nf
frequent_number=zeros(1,400);
number=j;
str='';
   if(number<nf)
      for i=index(number)+1:index(number+1)-1  %��number������������Խ��������
          str=[str,ex{i}];   %�ѵ�number�����������д����ַ���str��
      end
   end
   
   if (number==nf) %�������һ��������
      for i=index(nf)+1:size(ex,1)
          str=[str,ex{i}]; 
      end
   end
  
   h=1;
   for n=1:20
       for k=1:20
           Dipstr=strcat(indicator{n},indicator{k});  %��������壻ע����{}����������
           frequent_number(h)=size(strfind(str,Dipstr),2)/(size(str,2)-1); %size(str,2)-1��ʾ���������ʶ�����ĳ���
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




                            