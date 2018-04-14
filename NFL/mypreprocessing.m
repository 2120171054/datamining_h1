function data = mypreprocessing(analytic_mat, method)
%PREPROCESSING 对数据进行预处理，支持三种种方式（method取值为1~3）。
%   三种方式分别为：最高频值替代，属性相关关系，对象相似性。将缺失部分剔除是缺省方式，不在预处理函数中。

ATTRIBUTE_L = 1;
ATTRIBUTE_H = size(analytic_mat,2); % 

[m, n] = size(analytic_mat); % 输入矩阵的大小

standard_line = [analytic_mat(4, ATTRIBUTE_L: ATTRIBUTE_H)]; % 取出一行无缺失的样本，作为方法2插值的依据

switch(method)
    case 1 % 使用最高频值替代NaN
        for i = 1: m
            disp(i)
            for j = ATTRIBUTE_L: ATTRIBUTE_H 
                if(isnan(analytic_mat(i, j)))
                    analytic_mat(i, j) = max([analytic_mat(:, j)]); % 取本属性的最高频值
                end
            end
        end
        data = analytic_mat;
    case 2 % 使用最相关的属性补全NaN
        cor_mat = correlation_mat_attribute(analytic_mat); % 获得相关性矩阵，函数见下
        cor_size = size(cor_mat, 1); % 矩阵大小，正常情况下是方阵
        for i = 1: m
            disp(i)
            for j = ATTRIBUTE_L: ATTRIBUTE_H
                    if(isnan(analytic_mat(i, j)))
                    [~, index] = sort(cor_mat(j - ATTRIBUTE_L + 1, :));
                    index_list = fliplr(index); % sort升序，fliplr翻转，变成降序，得到参考的属性优先度列表
                    flag = 0; % 标识是否补全成功
                    for k = 1: cor_size
                        ref_attr = index_list(k); % 用于补全参考的属性
                        if(~isnan(analytic_mat(i, ref_attr)))
                            analytic_mat(i, j) = standard_line(j - ATTRIBUTE_L + 1) / standard_line(ref_attr) * ...
                                analytic_mat(i, ref_attr + ATTRIBUTE_L - 1); % 按比例补全（这不是最好的方法）
                            flag = 1;
                            break;
                        end
                    end
                    if(flag == 0)
                        disp(['Insert fail at row ', num2str(i), ' col ', num2str(j)]);
                        return ;
                    end
                    end
            end
        end
        data = analytic_mat;
    case 3 % 使用最相似的数据样本补全NaN，方式与上类似但略不同
        sim_mat = similarity_mat_sample(analytic_mat); % 获得相关性矩阵，函数见下
        sim_size = size(sim_mat, 1); % 矩阵大小，正常情况下是方阵
        for i = 1: m
            for j = ATTRIBUTE_L: ATTRIBUTE_H
                if(isnan(analytic_mat(i, j)))
                    [~, index_list] = sort(sim_mat(i, :));
                    flag = 0; % 标识是否补全成功
                    for k = 1: sim_size
                        ref_samp = index_list(k); % 用于补全参考的属性
                        if(~isnan(analytic_mat(ref_samp, j)))
                            analytic_mat(i, j) = analytic_mat(ref_samp, j); % 原样填上，补全
                            flag = 1;
                            break;
                        end
                    end
                    if(flag == 0)
                        disp(['Insert fail at row ', num2str(i), ' col ', num2str(j)]);
%                         return ;
                    end
                end
            end
        end
        data = analytic_mat;
    otherwise
        data = [];
        disp('Invalid method input.');
end

end

function cor_mat = correlation_mat_attribute(analytic_mat)
%CORRELATION_MAT_ATTRIBUTE 计算属性之间的相关性矩阵。
%   就是行和列都是属性，value(i,j)是属性i和属性j的相关性。它是对称矩阵

ATTRIBUTE_L = 1;
ATTRIBUTE_H = size(analytic_mat,2); 
COR_SIZE = ATTRIBUTE_H - ATTRIBUTE_L + 1; % 相关性矩阵的大小

cor_mat = ones(COR_SIZE, COR_SIZE); % 初始化相关性矩阵，由于要取最大相关性，初始为最小值（-1）
for i = ATTRIBUTE_L: ATTRIBUTE_H - 1
    for j = i + 1: ATTRIBUTE_H
        merge = [[analytic_mat(:, i)], [analytic_mat(:, j)]]; % 将待求相关系数的两列并起来
        [NaN_line, ~] = find(isnan(merge) == 1);
        merge(NaN_line, :) = []; % 删掉含有NaN的行以便正确求解相关系数
        cor_indx = i - ATTRIBUTE_L + 1;
        cor_indy = j - ATTRIBUTE_L + 1; % 相关性矩阵下标
        if ~isempty(merge)
          cor_mat(cor_indx, cor_indy) = corr(merge(:, 1), merge(:, 2)); % merge的两列即去除NaN的两属性，求相关系数
          cor_mat(cor_indy, cor_indx) = cor_mat(cor_indx, cor_indy); % 对称矩阵
        else
            cor_mat(cor_indx, cor_indy) = 0; % merge的两列即去除NaN的两属性，求相关系数
            cor_mat(cor_indy, cor_indx)=0;
        end
    end
end

end

function sim_mat = similarity_mat_sample(analytic_mat)
%SIMILARITY_MAT_SAMPLE 计算各个样本间的相似性。与correlation_mat_attribute类似。
%   此处相似性实际上求了欧几里得距离，因此越小越相似。其余与上述函数类似，不做多余的注释。

ATTRIBUTE_L = 1;
ATTRIBUTE_H = size(analytic_mat,2); 

SIM_SIZE = size(analytic_mat, 1); % 相似矩阵大小，与analytic_mat样本数一致

sim_mat = ones(SIM_SIZE, SIM_SIZE) * 999; % 初始化为最大距离
for i = 1: SIM_SIZE - 1
    for j = i + 1: SIM_SIZE
        merge = [[analytic_mat(i, ATTRIBUTE_L: ATTRIBUTE_H)]', ...
            [analytic_mat(j, ATTRIBUTE_L: ATTRIBUTE_H)]']; % 将两行样本转置合并为属性数x2的矩阵
        [NaN_line, ~] = find(isnan(merge) == 1);
        merge(NaN_line, :) = [];
        
        sim_mat(i, j) = norm(merge(:, 1) - merge(:, 2)); % 两样本的欧几里得距离
        sim_mat(j, i) = sim_mat(i, j); % 对称矩阵
    end
end

end