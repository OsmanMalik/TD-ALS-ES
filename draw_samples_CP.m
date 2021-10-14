function [samples, sqrt_probs] = draw_samples_CP(A, sketch, n, no_samp)
%draw_samples_CP Draws samples for CP design matrix as discussed in our
%paper
%
%samples = draw_samples_CP(A, sketch, n, no_samp) returns no_samp samples
%organized into an no_samp by N matrix, where N is the number of modes of
%the tensor being decomposed. The cell A should contain all the CP factor
%matrices in standard order, and the n-th factor matrix is the one being
%solved for. The n-th column of samples will just be NaN, since that index
%is not being sampled. 

N = length(A);

% Compute matrix Phi
[~, Sigma, V] = svd(sketch, 'econ');
temp = V*inv(Sigma);
Phi = temp*temp';

% Precompute the two terms in Eq (15) in paper
M1 = cell(1,N);
M2 = cell(1,N);
for j = 1:N
    if j ~= n
        M1{j} = khatrirao(A{j}.', A{j}.');
        M2{j} = A{j}.' * A{j};
    end
end

% Main loop for drawing all samples
samples = nan(no_samp, N); % Each row is a realization (i_j)_{j \neq n}
sqrt_probs = ones(no_samp, 1); % To store the square root of the probability of each drawn sample
if n == 1
    first_idx = 2;
else
    first_idx = 1;
end
first_idx_flag = true;
for samp = 1:no_samp
    % Compute P(i_m | (i_j)_{j < m, ~= n}) for each m (~=n) and draw i_m 
    for m = first_idx:N
        if m == first_idx && ~first_idx_flag
            continue
        end
        
        if m ~= n
            
            % Compute conditional probability vector
            common_terms = Phi(:);
            for j = 1:N
                if j < m && j ~= n
                    common_terms = common_terms .* M1{j}(:, samples(samp, j));
                elseif j > m && j ~= n
                    common_terms = common_terms .* M2{j}(:);
                end
            end
            common_terms = common_terms.';
            const = common_terms * M2{m}(:);
            common_terms = common_terms / const;
            prob_m = common_terms * M1{m};
            
            if first_idx_flag
                % Draw from the vector
                samples(:, m) = randsample(length(prob_m), no_samp, true, prob_m);
                
                % Update probability vector
                sqrt_probs = sqrt(prob_m(samples(:, m)));
                
                first_idx_flag = false;
            else
                % Draw from the vector
                samples(samp, m) = mt19937ar(prob_m);
                
                % Update probability vector
                sqrt_probs(samp) = sqrt_probs(samp) * sqrt(prob_m(samples(samp,m)));
            end
            
        end
    end
end

sqrt_probs = sqrt_probs';

end
