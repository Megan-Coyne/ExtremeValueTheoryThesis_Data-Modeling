%% Load data
T = readtable('data/FordStock.csv'); % adjust path if needed
T.date = datetime(T.date,'InputFormat','yyyy-MM-dd'); % adjust format
T.abs_RET = abs(T.RET);

%% POT + GPD for overall data
thresholdQuantile = 0.95;
threshold = quantile(T.abs_RET, thresholdQuantile);
exceedances = T.abs_RET(T.abs_RET > threshold) - threshold; % POT

% Fit GPD to exceedances
[paramEsts] = gpfit(exceedances);  % [shape k, scale sigma]
k = paramEsts(1);
sigma = paramEsts(2);
fprintf('Overall GPD Fit (POT > %.4f) - Shape (k): %.4f, Scale (sigma): %.4f\n', ...
    threshold, k, sigma);

% Plot histogram + GPD PDF
figure;
histogram(exceedances,50,'Normalization','pdf','FaceColor','skyblue'); hold on;
x = linspace(min(exceedances), max(exceedances), 1000);
plot(x, gppdf(x,k,sigma,0),'r-','LineWidth',2);
xlabel('Exceedances over threshold'); ylabel('Density');
title(sprintf('GPD Fit to Absolute Returns (Threshold = %.4f)', threshold));
legend('Exceedances','GPD Fit');
grid on;

%% POT + GPD Year-by-Year
years = [1990, 1995, 2000, 2005, 2010];
colors = {'g','orange','purple','brown','pink'};
fit_results = {}; 
fit_results(end+1,:) = {'Overall', num2str(k,'%.4f'), num2str(sigma,'%.4f'), num2str(threshold,'%.4f')};

figure; hold on;

for i = 1:length(years)
    year = years(i);
    data_year = T.abs_RET(year(T.date) == year);
    data_year = data_year(~isnan(data_year));
    
    if isempty(data_year)
        continue
    end
    
    % POT
    threshold_y = quantile(data_year, thresholdQuantile);
    exceedances_y = data_year(data_year > threshold_y) - threshold_y;
    
    % GPD fit
    [paramEsts_y] = gpfit(exceedances_y);
    k_y = paramEsts_y(1);
    sigma_y = paramEsts_y(2);
    fprintf('Year %d - Threshold: %.4f, Shape = %.4f, Scale = %.4f\n', ...
        year, threshold_y, k_y, sigma_y);
    
    % Plot PDF
    x_y = linspace(min(exceedances_y), max(exceedances_y), 1000);
    plot(x_y, gppdf(x_y, k_y, sigma_y, 0), 'LineWidth',2,'Color',colors{i}, ...
        'DisplayName', sprintf('%d GPD Fit (POT> %.4f)', year, threshold_y));
    
    fit_results(end+1,:) = {num2str(year), num2str(k_y,'%.4f'), num2str(sigma_y,'%.4f'), num2str(threshold_y,'%.4f')};
end

xlabel('Exceedances over threshold'); ylabel('Density');
title('GPD Fits Year-by-Year (POT)');
legend('show'); grid on;

%% Display MLE Table
f = figure('Position',[100 100 600 200]);
t = uitable(f,'Data',fit_results, ...
    'ColumnName',{'Dataset/Year','Shape (k)','Scale (\sigma)','Threshold (u)'},...
    'ColumnWidth',{100 100 100 100},'FontSize',12);
