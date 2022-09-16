
data {
  array[100] int<lower=1,upper=4> choice;  //arm choice
  array[100] int reward;  // reward
}

parameters {
  real logit_alpha; // learning rate
  real log_beta; // inverse temperature
}

model {
  vector[4] Q_values; // Q values for each arm
  vector[4] probs; // probabilities for each arm
 
  // priors
  logit_alpha ~ normal(0,1);
  log_beta ~ normal(-1,1);
  
  // initial Q values all zero
  Q_values = rep_vector(0,4); 
  
  // trial loop
  for (t in 1:100) {
      
    real alpha;
    real beta;
      
    // get softmax probabilities from Q_values
    beta = exp(log_beta);
    probs = softmax(beta * Q_values);
      
    // choose an arm based on probs
    choice[t] ~ categorical(probs);
      
    //update Q_values
    alpha = inv_logit(logit_alpha);
    Q_values[choice[t]] = Q_values[choice[t]] + alpha * (reward[t] - Q_values[choice[t]]);
      
  }
  
}

generated quantities {
  real alpha;
  real beta;
  
  alpha = inv_logit(logit_alpha);
  beta = exp(log_beta);
}
