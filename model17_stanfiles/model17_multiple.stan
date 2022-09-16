
data {
  int N;  // number of agents
  array[100, N] int reward; // reward
  array[100, N] int<lower=1,upper=4> choice; // arm choice  
}

parameters {
  real logit_alpha; // learning rate grand mean
  real log_beta; // inverse temperature grand mean
  
  matrix[2, N] z_i;  // matrix of uncorrelated z-values
  vector<lower=0>[2] sigma_i; // sd of parameters across individuals
  cholesky_factor_corr[2] Rho_i; // cholesky factor: multiply this matrix and its transpose to get correlation matrix
}

transformed parameters{
  matrix[2, N] v_i; // matrix of varying effects for each individual
  v_i = (diag_pre_multiply(sigma_i, Rho_i ) * z_i);
}

model {
  array[N] vector[4] Q_values; // Q values per agent
  vector[4] probs; // probabilities for each arm
 
  // priors
  logit_alpha ~ normal(0,1);
  log_beta ~ normal(-1,1);
  
  to_vector(z_i) ~ normal(0,1);
  sigma_i ~ exponential(1);
  Rho_i ~ lkj_corr_cholesky(4);
  
  // initial Q values all zero
  for (i in 1:N) Q_values[i] = rep_vector(0,4); 
  
  // trial loop
  for (t in 1:100) {
    
    // agent loop
    for (i in 1:N) {
      
      real alpha;  // v_i parameter 1
      real beta;  // v_i parameter 2
      
      // get softmax probabilities from Q_values
      beta = exp(log_beta + v_i[2,i]);
      probs = softmax(beta * Q_values[i]);
      
      // choose an arm based on probs
      choice[t,i] ~ categorical(probs);
      
      //update Q_values
      alpha = inv_logit(logit_alpha + v_i[1,i]);
      Q_values[i,choice[t,i]] = Q_values[i,choice[t,i]] + alpha * (reward[t,i] - Q_values[i,choice[t,i]]);
      
    }
  
  }
  
}

generated quantities {
  real alpha;
  real beta;
  
  alpha = inv_logit(logit_alpha);
  beta = exp(log_beta);
}
