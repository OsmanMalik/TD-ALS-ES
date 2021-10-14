function SA = TensorSketch(A, J, varargin)
%TENSORSKETCH Computes the TensorSketch of a matrix Khatri-Rao product
%
%SA = TensorSketch(A, J) returns the TensorSketch of the Khatri-Rao 
%product of the matrices in A using a target sketch dimension J. A
%should be a (row or column) cell containing the matrices. The matrices 
%in A can be either dense or sparse: The appropriate countSketch 
%function will be used in each case.
%
%SA = TensorSketch(___, 'h', h, 's', s) allows the user to provide the
%cells with the hash functions manually. This is helpful if e.g. the same
%realization of a TensorSketch needs to be applied twice via seperate calls
%to the function.

%% Handle optional inputs

params = inputParser;
addParameter(params, 'h', nan);
addParameter(params, 's', nan);
parse(params, varargin{:});

h = params.Results.h;
s = params.Results.s;

%% Computations

N = length(A);
R = size(A{1}, 2);
Acs = cell(size(A)); % To store CountSketch of each matrix in A. FFT and transpose are also applied.
P = ones(J, R);

% Define "target row" hash functions if they haven't been provided as
% inputs
if ~iscell(h)
    h = cell(N, 1);
    for n = 1:N
        h{n} = randi(J, size(A{n}, 1), 1);
    end
end

% Define "sign" hash functions if they haven't been provided as inputs
if ~iscell(s)
    s = cell(N, 1);
    for n = 1:N
        s{n} = randi(2, size(A{n}, 1), 1)*2-3;
    end
end

% Perform computations
for n = 1:N
    if issparse(A{n})
        Acs{n} = fft(countSketch_sparse(A{n}.', int64(h{n}), J, s{n}).');
    else
        Acs{n} = fft(countSketch(A{n}.', int64(h{n}), J, s{n}, 1).');
    end
    P = P.*Acs{n};
end

SA = ifft(P);

end
