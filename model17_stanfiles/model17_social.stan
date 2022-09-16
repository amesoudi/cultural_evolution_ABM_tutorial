
data {
  int N;  // number of agents
  array[100, N] int reward; // reward
  array[100, N] int<lower=1,upper=4> choice; // arm choice  
  array[100, 4] int<lower=0,upper=N> n; // number of agents picking each arm
}

parameters {
  real logit_alpha; // learning rate grand mean
  real log_beta; // inverse temperature grand mean
  real logit_s; // social learning prob grand mean
  real log_f; // conformity parameter grand mean
  
  matrix[4,N] z_i;  // matrix of uncorrelated z-values
  vector<lower=0>[4] sigma_i; // sd of parameters across individuals
  cholesky_factor_corr[4] Rho_i; // cholesky factor
}

transformed parameters{
  matrix[4,N] v_i; // matrix of varying effects for each individual
  v_i = (diag_pre_multiply(sigma_i, Rho_i ) * z_i);
}

model {
  array[N] vector[4] Q_values; // Q values per agent
  vector[4] probs; // probabilities for each arm
  vector[4] p_RL; // reinforcement learning probabilities
  vector[4] p_SL; // social learning probabilities
 
  // priors
  logit_alpha ~ normal(0,1);
  log_beta ~ normal(-1,1);
  logit_s ~ normal(0,1);
  log_f ~ normal(-1,1);
  
  to_vector(z_i) ~ normal(0,1);
  sigma_i ~ exponential(1);
  Rho_i ~ lkj_corr_cholesky(4);
  
  // initial Q values all zero
  for ( i in 1:N ) Q_values[i] = rep_vector(0,4); 
  
  // trial loop
  for (t in 1:100) {
    
    // agent loop
    for (i in 1:N) {
      
      real alpha;  // v_i parameter 1
      real beta;  // v_i parameter 2
      real s;  // v_i parameter 3
      real f;  // v_i parameter 4
      
      // get asocial softmax probabilities from Q_values
      beta = exp(log_beta + v_i[2,i]);
      p_RL = softmax(beta * Q_values[i]);
      
      if (t == 1) {
        
        // first generation has no social information
        probs = p_RL;
        
      } else {
        
        // from t=2 onwards, do conformity according to f
        
        f = exp(log_f + v_i[4,i]);
        for (arm in 1:4) p_SL[arm] = n[t-1,arm]^f;
        p_SL = p_SL / sum(p_SL);

        //update probs by combining p_RL and p_SL according to s
        s = inv_logit(logit_s + v_i[3,i]);
        probs = (1-s) * p_RL + s * p_SL;
      
      }
        
      // choose an arm based on probs
      choice[t,i] ~ categorical(probs);
    
      //update Q values
      alpha = inv_logit(logit_alpha + v_i[1,i]);
      Q_values[i,choice[t,i]] = Q_values[i,choice[t,i]] + alpha * (reward[t,i] - Q_values[i,choice[t,i]]);
      
    }
    
  }
  
}

generated quantities {
  real alpha;
  real beta;
  real s;
  real f;
  
  alpha = inv_logit(logit_alpha);
  beta = exp(log_beta);
  s = inv_logit(logit_s);
  f = exp(log_f);
}
