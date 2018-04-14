%%
%visualization
function myvisualization(path,totaldata,attribute,ATTRIBUTES_Number)
%VISUALIZATION 可视化数据。如果data未经预处理会有NaN，因此要先考虑剔除NaN的情况，再分别可视化。
%   绘制7个属性的直方图、QQ图、盒图

numofplot = 6; %根据此数值来设置一个figure绘图个数
if ~exist(path,'dir')
    mkdir(path);
end
data=zeros(size(totaldata,1),length(ATTRIBUTES_Number));
for m=1:length(ATTRIBUTES_Number)
    data(:,m)=totaldata(:,find(strcmp(attribute,ATTRIBUTES_Number(m))));
end
for i = 1:length(ATTRIBUTES_Number)
    if mod(i,numofplot) ==1
        fig = figure('visible','off','NumberTitle','off','Name',['histogram ' num2str(floor((i-1)/6))]); 
    end
    % 删掉含有NaN的行
    temp_data = data(:,i);
    [NaN_line, ~] = find(isnan(temp_data) == 1);
    temp_data(NaN_line,:) = [];
    % 画出属性的直方图
    subplot(2,3, mod(i-1,numofplot)+1);
    histogram(temp_data,50);
    title(ATTRIBUTES_Number(i));
    if mod(i,numofplot) == 0
        saveas(fig,[path 'histogram_' num2str(floor((i-1)/6)) '.jpg']);
    end
end

% 新建QQ图窗口，因为不知道怎么把subplot放到不同figure里只能重写一遍
for i = 1:length(ATTRIBUTES_Number)
    % 删掉含有NaN的行
    if mod(i,numofplot) ==1
        fig2=figure('visible','off','NumberTitle','off','Name',['qqplot ' num2str(floor((i-1)/numofplot))]);
    end
    temp_data = data(:,i);
    [NaN_line, ~] = find(isnan(temp_data) == 1);
    temp_data(NaN_line, :) = [];
    % 画出属性的QQ图
    subplot(2, 3, mod(i-1,numofplot)+1);
    qqplot(temp_data);
    title(ATTRIBUTES_Number(i));
    set(gca, 'xlabel', [], 'ylabel', []); % 取消横纵轴名
    if mod(i,numofplot) == 0
        saveas(fig2,[path 'qqplot_' num2str(floor((i-1)/numofplot)) '.jpg']);
    end
end


figure; % 新建盒图窗口
for i = 1:length(ATTRIBUTES_Number)
    if mod(i,numofplot) ==1
        fig3=figure('visible','off','NumberTitle','off','Name',['boxplot ' num2str(floor((i-1)/numofplot))]); 
    end
    % 删掉含有NaN的行
    temp_data = data(:,i);
    [NaN_line, ~] = find(isnan(temp_data) == 1);
    temp_data(NaN_line, :) = [];
    % 画出属性的盒图
    subplot(2, 3, mod(i-1,numofplot)+1);
    boxplot(temp_data);
    title(ATTRIBUTES_Number(i));
    set(gca, 'xticklabel', []); % 取消横轴名
    if mod(i,numofplot) == 0
        saveas(fig3,[path 'boxplot_' num2str(floor((i-1)/numofplot)) '.jpg']);
    end
end

end
