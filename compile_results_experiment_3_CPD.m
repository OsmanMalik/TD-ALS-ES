% A simple script to compile and print the outputs from experiment_3.m for
% all the different methods we consider. This was used to put together the
% CPD results that appear in Table 5 in our paper.

load experiment_3_cp_als_results
fprintf('CP-ALS:\n')
fprintf('Mean time: %.1f\n', mean(TIME))
fprintf('Mean error: %.2f\n', mean(ER))
fprintf('Mean accuracy: %.1f\n\n', 100*mean(ACC))

load experiment_3_cpd_als_results
fprintf('CPD-ALS:\n')
fprintf('Mean time: %.1f\n', mean(TIME))
fprintf('Mean error: %.2f\n', mean(ER))
fprintf('Mean accuracy: %.1f\n\n', 100*mean(ACC))

load experiment_3_cpd_minf_results
fprintf('CPD-MINF:\n')
fprintf('Mean time: %.1f\n', mean(TIME))
fprintf('Mean error: %.2f\n', mean(ER))
fprintf('Mean accuracy: %.1f\n\n', 100*mean(ACC))

load experiment_3_cpd_nls_results
fprintf('CP-NLS:\n')
fprintf('Mean time: %.1f\n', mean(TIME))
fprintf('Mean error: %.2f\n', mean(ER))
fprintf('Mean accuracy: %.1f\n\n', 100*mean(ACC))

load experiment_3_cp_arls_lev_results
fprintf('CP-ARLS-LEV:\n')
fprintf('Mean time: %.1f\n', mean(TIME))
fprintf('Mean error: %.2f\n', mean(ER))
fprintf('Mean accuracy: %.1f\n\n', 100*mean(ACC))

load experiment_3_cp_als_es_results
fprintf('CP-ALS-ES:\n')
fprintf('Mean time: %.1f\n', mean(TIME))
fprintf('Mean error: %.2f\n', mean(ER))
fprintf('Mean accuracy: %.1f\n\n', 100*mean(ACC))
