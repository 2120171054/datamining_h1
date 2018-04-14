%%
%visualization
function myvisualization(totaldata,attribute)
%VISUALIZATION 可视化数据。如果data未经预处理会有NaN，因此要先考虑剔除NaN的情况，再分别可视化。
%   绘制6个属性的直方图、QQ图、盒图

figure; % 新建直方图窗口

ATTRIBUTES_Number = {'Number of Existing Stories'; 'Number of Proposed Stories'; 'Estimated Cost'; 'Revised Cost'; 'Existing Units'; 'Proposed Units'}; 
data=zeros(198900,length(ATTRIBUTES_Number));
for m=1:length(ATTRIBUTES_Number)
    data(:,m)=totaldata(:,find(strcmp(attribute,ATTRIBUTES_Number(m))));
end
for i = 1:length(ATTRIBUTES_Number)
    % 删掉含有NaN的行
    temp_data = data(:,i);
    [NaN_line, ~] = find(isnan(temp_data) == 1);
    temp_data(NaN_line,:) = [];
    % 画出属性的直方图
    subplot(2,3, i);
    histogram(temp_data,50);
    title(ATTRIBUTES_Number(i));
end

figure; % 新建QQ图窗口，因为不知道怎么把subplot放到不同figure里只能重写一遍
for i = 1:length(ATTRIBUTES_Number)
    % 删掉含有NaN的行
    temp_data = data(:,i);
    [NaN_line, ~] = find(isnan(temp_data) == 1);
    temp_data(NaN_line, :) = [];
    % 画出属性的QQ图
    subplot(2, 3, i);
    qqplot(temp_data);
    title(ATTRIBUTES_Number(i));
    set(gca, 'xlabel', [], 'ylabel', []); % 取消横纵轴名
end


figure; % 新建盒图窗口
for i = 1:length(ATTRIBUTES_Number)
    % 删掉含有NaN的行
    temp_data = data(:,i);
    [NaN_line, ~] = find(isnan(temp_data) == 1);
    temp_data(NaN_line, :) = [];
    % 画出属性的盒图
    subplot(2, 3, i);
    boxplot(temp_data);
    title(ATTRIBUTES_Number(i));
    set(gca, 'xticklabel', []); % 取消横轴名
end

end
