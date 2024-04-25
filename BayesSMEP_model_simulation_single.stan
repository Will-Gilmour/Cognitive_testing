
data {
  int<lower=1> nTrials;
  real<lower=0, upper=100> reward[nTrials,4];    
  real<lower=0, upper=3> beta;
  real phi;
  real persev;

  }

transformed data {
  real<lower=0, upper=100> Q1;
  real<lower=0> sig1;
  real<lower=0> sigO;
  real<lower=0> sigD;
  real<lower=0,upper=1> decay;
  real<lower=0, upper=100> decay_center;
  
  // random walk parameters 
  Q1   = 50.0;        // prior belief mean reward value trial 1
  sig1 = 4.0;         // prior belief variance trial 1
  sigO = 4.0;         // observation variance
  sigD = 2.8;         // diffusion variance
  decay = 0.9836;     // decay parameter
  decay_center = 50;  // decay center
}


generated quantities {
    int choice[nTrials];
    real reward_obt[nTrials];

    vector[4] Q;   // value (Q)
    vector[4] sig; // sigma
    vector[4] eb;  // exploration bonus
    vector[4] pb;  // perseveration bonus
    real pe;       // prediction error
    real Kgain;    // Kalman gain

    Q   = rep_vector(Q1, 4);
    sig = rep_vector(sig1, 4);

    for (t in 1:nTrials) {
        eb = phi * sig; // update Exploration bonus
        pb = rep_vector(0.0, 4); // Reset perseveration bonus

        if (t>1) {
            pb[choice[t-1]] = persev; // Bandit last chosen gets bonus
        }

        choice[t] = categorical_logit_rng( beta * (Q + eb + pb )); // generate action probabilities and selection an action based on this
        reward_obt[t] = reward[t,choice[t]]; // reward obtained based on choice simulated

        pe    = reward_obt[t] - Q[choice[t]];                       // prediction error
        Kgain = sig[choice[t]]^2 / (sig[choice[t]]^2 + sigO^2); // Kalman gain

        Q[choice[t]]   = Q[choice[t]] + Kgain * pe;             // value/mu updating (learning)
        sig[choice[t]] = sqrt( (1-Kgain) * sig[choice[t]]^2 );  // sigma updating

        Q = decay * Q + (1-decay) * decay_center;
        for (j in 1:4) {
          sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
        }
    }
}  
