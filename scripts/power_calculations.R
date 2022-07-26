library(DeclareDesign)

design <-
  declare_model(
    N = 100, 
    potential_outcomes(Y ~ rbinom(N, size = 1, prob = 0.5 + 0.05 * Z))
  ) +
  declare_inquiry(ATE = 0.05) +
  declare_assignment(Z = complete_ra(N, m = 50)) +
  declare_measurement(Y = reveal_outcomes(Y ~ Z)) + 
  declare_estimator(Y ~ Z, model = lm_robust, inquiry = "ATE")

diagnosands <-
  declare_diagnosands(bias = mean(estimate - estimand),
                      power = mean(p.value <= 0.05))

diagnosis <- diagnose_design(design, diagnosands = diagnosands)