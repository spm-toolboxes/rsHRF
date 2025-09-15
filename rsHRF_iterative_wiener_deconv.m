function [xhat, iter, params] = rsHRF_iterative_wiener_deconv(y, h, varargin)
% Iterative Wiener-like deconvolution with wavelet-based noise estimation
%
% Inputs:
%   y - observed signal (N×1)
%   h - HRF or system impulse response (Nh×1)
%
% Parameters (Name-Value pairs):
%   'MaxIter' - maximum iterations (default 50)
%   'Tol'     - convergence criterion (default 1e-4)
%   'TR'      - sampling interval (seconds)
%   'Mode'    - 'rest' (default) or 'task'
%   'Smooth'  - temporal smoothing window (points)
%   'LowPass' - low-pass cutoff frequency (Hz)

% ---------------- Parameter parsing ----------------
p = inputParser;
addParameter(p, 'MaxIter', 50);
addParameter(p, 'Tol', 1e-4);
addParameter(p, 'TR', []);
addParameter(p, 'Mode', 'rest');
addParameter(p, 'Smooth', []);
addParameter(p, 'LowPass', []);
parse(p, varargin{:});
opts = p.Results;

% ---------------- Preprocessing ----------------
y = y - nanmean(y);  % mean-center to remove DC offset
N = length(y);
nh = length(h);
if nh < N
    h = [h; zeros(N-nh,1)];
elseif nh > N
    h = h(1:N);
end

% Sampling rate
fs = 1;
if ~isempty(opts.TR)
    fs = 1/opts.TR;
end
nyquist = fs/2;

% Auto-recommend Smooth and LowPass
if isempty(opts.Smooth) || isempty(opts.LowPass)
    switch lower(opts.Mode)
        case 'rest'
            smoothRec = max(round(4/opts.TR),3);
            lowpassRec = min(0.2,0.8*nyquist);
        case 'task'
            smoothRec = max(round(2/opts.TR),2);
            lowpassRec = min(0.35,0.9*nyquist);
        otherwise
            error('Unknown Mode: %s', opts.Mode);
    end
end
if isempty(opts.Smooth), opts.Smooth = smoothRec; end
if isempty(opts.LowPass), opts.LowPass = lowpassRec; end

% ---------------- FFT preprocessing ----------------
Hfft = fft(h);
Yfft = fft(y);

% Initial estimate
xhat = y;
Pxx = abs(Yfft).^2;

% Initial noise estimate
[c,l] = wavedec(y,1,'db2');
sigma = wnoisest(c,l,1);
Nf = sigma^2 * N;

% ---------------- Iterative process ----------------
for iter = 1:opts.MaxIter
    % Wiener-like update
    M = (conj(Hfft).*Pxx.*Yfft) ./ (abs(Hfft).^2.*Pxx + Nf);
    PxxY = (Pxx .* Nf) ./ (abs(Hfft).^2 .* Pxx + Nf);
    Pxx_new = PxxY + abs(M).^2;

    WienerFilterEst = (conj(Hfft).*Pxx_new) ./ ((abs(Hfft).^2.*Pxx_new) + Nf);
    xhat_new = real(ifft(WienerFilterEst .* Yfft));

    % ---------------- Smoothing ----------------
    if opts.Smooth > 1
        g = gausswin(opts.Smooth);
        g = g / sum(g);
        xhat_new = conv(xhat_new, g, 'same');
    end

    % ---------------- Low-pass ----------------
    if opts.LowPass < nyquist
        f = (0:N-1)'/N*fs;
        Xf = fft(xhat_new);
        Xf(f > opts.LowPass) = 0;
        xhat_new = real(ifft(Xf));
    end

    % ---------------- Dynamic noise update ----------------
    residual = y - conv(xhat_new,h,'same');
    [c,l] = wavedec(residual,1,'db2');
    sigma = wnoisest(c,l,1);
    Nf = sigma^2 * N;

    % ---------------- Convergence ----------------
    if norm(xhat_new - xhat)/norm(xhat) < opts.Tol
        xhat = xhat_new;
        break;
    end

    xhat = xhat_new;
    Pxx = Pxx_new;
end

params = opts;
end
