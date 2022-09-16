
data {
  array[100] int<lower=1,upper=4> choice;  //arm choice
  array[100] int reward;  // reward
}

parameters {
  real<lower = 0, upper = 1> alpha; // learning rate
  real<lower = 0> beta; // inverse temperature
}

model {
  vector[4] Q_values; // Q values for each arm
  vector[4] probs; // probabilities for each arm
 
  // priors
  alpha ~ beta(1,1);
  beta ~ normal(0,3);
  
  // initial Q values all zero
  Q_values = rep_vector(0,4); 
  
  // trial loop
  for (t in 1:100) {
      
    // get softmax probabilities from Q_values
    probs = softmax(beta * Q_values);
      
    // choose an arm based on probs
    choice[t] ~ categorical(probs);
      
    //update Q_values
    Q_values[choice[t]] = Q_values[choice[t]] + alpha * (reward[t] - Q_values[choice[t]]);
      
  }
  
}
