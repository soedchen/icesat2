function [fitresult, gof] = createFit1(model, insitu)
%CREATEFIT1(MODEL,INSITU)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : model
%      Y Output: insitu
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 15-Aug-2021 15:02:10 自动生成


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( model, insitu );

% Set up fittype and options.
ft = fittype( 'poly1' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'off';

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
% figure( 'Name', 'untitled fit 1' );
% h = plot( fitresult, xData, yData );
% legend( h, 'insitu vs. model', 'untitled fit 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% % Label axes
% xlabel( 'model', 'Interpreter', 'none' );
% ylabel( 'insitu', 'Interpreter', 'none' );
% grid on


