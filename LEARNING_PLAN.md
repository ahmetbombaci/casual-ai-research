# Causal AI Learning Plan: 2 Hours Daily

## Overview

This structured learning plan is designed for **2 hours of daily study** over **12 weeks** (approximately 3 months). The plan takes you from foundational concepts to practical implementation, with hands-on projects and incremental complexity.

**Total Time Commitment:** ~168 hours over 84 days
**Schedule:** 2 hours/day, 6 days/week (1 rest day for review/catch-up)

---

## Learning Philosophy

### Daily Session Structure (2 hours)
- **Theory & Reading (45 min):** Core concepts, reading, watching lectures
- **Hands-on Practice (60 min):** Coding exercises, implementing examples
- **Reflection & Notes (15 min):** Summarize learnings, document challenges, plan next session

### Weekly Rhythm
- **Days 1-5:** New content and practice
- **Day 6:** Review, consolidation, mini-project work
- **Day 7:** Rest/optional catch-up

---

## Phase 1: Foundations (Weeks 1-3)
**Goal:** Build intuition about causality and understand core concepts

### Week 1: Introduction to Causal Thinking

#### Day 1: Correlation vs. Causation
- **Theory (45 min):**
  - Read: Introduction to Causal Inference (Brady Neal) - Chapter 1
  - Watch: "Correlation doesn't imply causation" examples
  - Understand: Simpson's Paradox

- **Practice (60 min):**
  - Work through Simpson's Paradox examples with real data
  - Create visualizations showing how correlations can mislead
  - Dataset: Use UCI Berkeley Admissions data

- **Reflection (15 min):**
  - Write down 3 examples from your domain where correlation ≠ causation
  - Note: What makes identifying causation difficult?

#### Day 2: Introduction to DAGs
- **Theory (45 min):**
  - Read: Brady Neal Chapter 2 - sections on DAGs
  - Learn: Nodes, edges, paths, and basic graph concepts
  - Understand: The difference between statistical and causal models

- **Practice (60 min):**
  - Install DoWhy: `pip install dowhy`
  - Draw your first DAGs using networkx
  - Create 5 simple causal graphs for familiar scenarios

- **Reflection (15 min):**
  - What makes a good causal graph?
  - Sketch DAG for a problem you're interested in

#### Day 3: Confounding and Backdoor Paths
- **Theory (45 min):**
  - Read: Brady Neal Chapter 2 - confounding section
  - Understand: Confounders, mediators, colliders
  - Learn: What is a backdoor path?

- **Practice (60 min):**
  - Implement examples with confounding in Python
  - Use simulated data to show how confounders create spurious correlations
  - Practice identifying confounders in DAGs

- **Reflection (15 min):**
  - Can you identify confounders in your own work problems?
  - What happens when you don't adjust for confounders?

#### Day 4: The Backdoor Criterion
- **Theory (45 min):**
  - Read: Brady Neal Chapter 2 - backdoor criterion
  - Understand: How to block backdoor paths
  - Learn: Valid adjustment sets

- **Practice (60 min):**
  - Use DoWhy to identify backdoor paths
  - Practice finding valid adjustment sets
  - Work through examples with multiple confounders

- **Reflection (15 min):**
  - How do you know if an adjustment set is valid?
  - Practice explaining backdoor criterion in your own words

#### Day 5: d-separation and Conditional Independence
- **Theory (45 min):**
  - Read: Brady Neal Chapter 2 - d-separation
  - Understand: Rules of d-separation
  - Learn: How graph structure implies independence

- **Practice (60 min):**
  - Implement d-separation tests in Python
  - Use causal-learn or DoWhy for conditional independence testing
  - Create complex DAGs and test d-separation

- **Reflection (15 min):**
  - Why is d-separation important for causal inference?
  - What are the three patterns that block/unblock paths?

#### Day 6: Week 1 Review & Mini-Project
- **Review (30 min):**
  - Revisit difficult concepts from Week 1
  - Go through your notes and questions

- **Mini-Project (90 min):**
  - **Project:** Analyze a real-world confounding scenario
  - Choose dataset (e.g., Titanic, or simple observational study)
  - Draw the DAG
  - Identify confounders
  - Compare naive vs. adjusted estimates
  - Document findings in a Jupyter notebook

---

### Week 2: Structural Causal Models and Interventions

#### Day 1: Structural Causal Models (SCMs)
- **Theory (45 min):**
  - Read: Brady Neal Chapter 3 - SCMs
  - Understand: Structural equations
  - Learn: How SCMs formalize causal relationships

- **Practice (60 min):**
  - Define SCMs in Python for simple systems
  - Simulate data from SCMs
  - Understand how changing equations changes data distribution

- **Reflection (15 min):**
  - How do SCMs differ from statistical models?
  - Write SCM for a problem you're studying

#### Day 2: Interventions and do-operator
- **Theory (45 min):**
  - Read: Brady Neal Chapter 3 - interventions
  - Understand: Difference between seeing P(Y|X) and doing P(Y|do(X))
  - Learn: Graph surgery/mutilation

- **Practice (60 min):**
  - Implement interventions using DoWhy
  - Compare observational vs. interventional distributions
  - Simulate interventions on SCMs

- **Reflection (15 min):**
  - Why can't we compute interventions from observations alone?
  - Give examples where P(Y|X) ≠ P(Y|do(X))

#### Day 3: Introduction to do-calculus
- **Theory (45 min):**
  - Read: Brady Neal Chapter 3 - do-calculus introduction
  - Understand: The three rules at high level
  - Learn: When we can identify causal effects from observational data

- **Practice (60 min):**
  - Work through do-calculus examples
  - Use DoWhy's identify_effect() method
  - Practice with simple graphs

- **Reflection (15 min):**
  - What makes a causal effect identifiable?
  - When do we need experiments vs. observational data?

#### Day 4: Front-door Criterion
- **Theory (45 min):**
  - Read: Brady Neal - front-door criterion section
  - Understand: Alternative to backdoor adjustment
  - Learn: When front-door is useful

- **Practice (60 min):**
  - Implement front-door adjustment
  - Compare with backdoor adjustment
  - Work through smoking example (cigarettes → tar → cancer)

- **Reflection (15 min):**
  - When is front-door better than backdoor?
  - Can you think of scenarios where front-door applies?

#### Day 5: Counterfactuals
- **Theory (45 min):**
  - Read: Brady Neal Chapter 4 - counterfactuals
  - Understand: The three levels of causal hierarchy
  - Learn: How to compute counterfactuals from SCMs

- **Practice (60 min):**
  - Implement counterfactual reasoning in DoWhy
  - Work through examples: "What if I had done X instead?"
  - Practice on synthetic data

- **Reflection (15 min):**
  - How are counterfactuals different from interventions?
  - Why can't counterfactuals always be computed from data?

#### Day 6: Week 2 Review & Mini-Project
- **Review (30 min):**
  - Consolidate understanding of SCMs, interventions, do-calculus

- **Mini-Project (90 min):**
  - **Project:** Build an SCM for a multi-variable system
  - Define structural equations
  - Simulate observational data
  - Perform interventions
  - Compute counterfactuals
  - Create visualizations showing differences
  - Document in Jupyter notebook

---

### Week 3: Potential Outcomes Framework

#### Day 1: Introduction to Potential Outcomes
- **Theory (45 min):**
  - Read: Brady Neal Chapter 5 - potential outcomes introduction
  - Understand: Y(0) and Y(1) notation
  - Learn: Fundamental problem of causal inference

- **Practice (60 min):**
  - Simulate potential outcomes framework
  - Understand selection bias
  - Calculate observed vs. true treatment effects

- **Reflection (15 min):**
  - How do potential outcomes relate to SCMs?
  - Why can we never observe both Y(0) and Y(1) for same unit?

#### Day 2: Treatment Effects (ATE, ATT, CATE)
- **Theory (45 min):**
  - Read: Brady Neal Chapter 5 - treatment effects
  - Understand: ATE, ATT, ATC, CATE, ITE
  - Learn: When each estimand is appropriate

- **Practice (60 min):**
  - Calculate different treatment effects from simulated data
  - Use CausalML library
  - Understand heterogeneous treatment effects

- **Reflection (15 min):**
  - When would you care about CATE vs. ATE?
  - How does heterogeneity affect decision-making?

#### Day 3: Assumptions for Identification
- **Theory (45 min):**
  - Read: Brady Neal Chapter 5 - assumptions (ignorability, positivity, SUTVA)
  - Understand: When treatment effects are identifiable
  - Learn: How assumptions can fail

- **Practice (60 min):**
  - Check assumptions in real datasets
  - Visualize overlap and positivity violations
  - Test for confounding

- **Reflection (15 min):**
  - Which assumption is most commonly violated?
  - How can you test assumptions in practice?

#### Day 4: Randomized Experiments vs. Observational Studies
- **Theory (45 min):**
  - Read: Brady Neal Chapter 5 - randomization
  - Understand: Why randomization solves confounding
  - Learn: Limitations of experiments

- **Practice (60 min):**
  - Simulate RCT vs. observational study
  - Show how randomization achieves balance
  - Compare estimates under both scenarios

- **Reflection (15 min):**
  - When are experiments not feasible?
  - What can go wrong even in randomized trials?

#### Day 5: Connecting DAGs and Potential Outcomes
- **Theory (45 min):**
  - Read: Brady Neal Chapter 5 - connection to DAGs
  - Understand: Two frameworks are complementary
  - Learn: When to use each framework

- **Practice (60 min):**
  - Express potential outcomes assumptions as DAGs
  - Express DAG assumptions in potential outcomes notation
  - Work through examples in both frameworks

- **Reflection (15 min):**
  - What are strengths of each framework?
  - When might you prefer one over the other?

#### Day 6: Week 3 Review & Phase 1 Project
- **Review (30 min):**
  - Consolidate all Phase 1 concepts
  - Review your notes from weeks 1-3

- **Phase 1 Project (90 min):**
  - **Project:** Comprehensive causal analysis
  - Choose an interesting observational dataset
  - Draw the causal DAG based on domain knowledge
  - Formulate causal question
  - State assumptions (identify confounders)
  - Estimate treatment effect using multiple methods
  - Discuss limitations and violations
  - Create presentation-ready notebook

---

## Phase 2: Estimation Methods (Weeks 4-6)
**Goal:** Learn practical methods for estimating causal effects

### Week 4: Basic Estimation Methods

#### Day 1: Regression for Causal Inference
- **Theory (45 min):**
  - Read: Brady Neal Chapter 6 - regression adjustment
  - Understand: When regression gives causal effects
  - Learn: Limitations of regression

- **Practice (60 min):**
  - Implement regression adjustment in Python
  - Compare with and without proper adjustment
  - Identify when regression fails (non-linearity, misspecification)

- **Reflection (15 min):**
  - Why isn't regression always enough?
  - What assumptions does regression make?

#### Day 2: Matching Methods
- **Theory (45 min):**
  - Read: Causal Inference What If - Chapter on matching
  - Understand: Exact matching, coarsened exact matching
  - Learn: When matching is appropriate

- **Practice (60 min):**
  - Implement exact matching using Python
  - Use libraries: CausalML or DoWhy
  - Compare matched vs. unmatched estimates

- **Reflection (15 min):**
  - What are limitations of exact matching?
  - How do you handle continuous confounders?

#### Day 3: Propensity Score Basics
- **Theory (45 min):**
  - Read: Brady Neal Chapter 7 - propensity scores
  - Understand: What is propensity score
  - Learn: Balancing property

- **Practice (60 min):**
  - Estimate propensity scores using logistic regression
  - Visualize propensity score distributions
  - Check balance before/after adjustment

- **Reflection (15 min):**
  - Why does propensity score work?
  - What information is lost when using propensity scores?

#### Day 4: Propensity Score Matching
- **Theory (45 min):**
  - Read: Brady Neal Chapter 7 - PS matching
  - Understand: Nearest neighbor, caliper matching
  - Learn: How to check balance

- **Practice (60 min):**
  - Implement PS matching using CausalML
  - Try different matching algorithms
  - Assess covariate balance using standardized differences

- **Reflection (15 min):**
  - How do you choose matching parameters?
  - What to do with unmatched units?

#### Day 5: Inverse Probability Weighting (IPW)
- **Theory (45 min):**
  - Read: Brady Neal Chapter 7 - IPW
  - Understand: How weighting creates pseudo-population
  - Learn: Stabilized weights

- **Practice (60 min):**
  - Implement IPW estimator
  - Compare IPW with matching
  - Visualize how weights work

- **Reflection (15 min):**
  - When might IPW be preferred over matching?
  - What problems can arise with extreme weights?

#### Day 6: Week 4 Review & Comparison Project
- **Review (30 min):**
  - Compare all estimation methods learned

- **Mini-Project (90 min):**
  - **Project:** Method comparison study
  - Use a benchmark dataset (e.g., Lalonde dataset)
  - Apply all methods: regression, matching, PS matching, IPW
  - Compare estimates and standard errors
  - Discuss which method works best and why
  - Document trade-offs

---

### Week 5: Advanced Estimation Methods

#### Day 1: Doubly Robust Methods
- **Theory (45 min):**
  - Read: Brady Neal Chapter 7 - doubly robust estimation
  - Understand: Why "doubly robust"
  - Learn: AIPW (Augmented IPW)

- **Practice (60 min):**
  - Implement AIPW using EconML
  - Compare with IPW and regression
  - Show double robustness property

- **Reflection (15 min):**
  - Why is double robustness desirable?
  - What are computational trade-offs?

#### Day 2: Instrumental Variables - Part 1
- **Theory (45 min):**
  - Read: Brady Neal Chapter 8 - IV basics
  - Understand: What is an instrument
  - Learn: Three IV assumptions

- **Practice (60 min):**
  - Work through classic IV examples (returns to education)
  - Implement 2SLS regression
  - Test instrument strength

- **Reflection (15 min):**
  - Why are good instruments hard to find?
  - How do you validate IV assumptions?

#### Day 3: Instrumental Variables - Part 2
- **Theory (45 min):**
  - Read: Brady Neal Chapter 8 - LATE and IV interpretation
  - Understand: Local Average Treatment Effect
  - Learn: Compliers, never-takers, always-takers

- **Practice (60 min):**
  - Implement IV using DoWhy and EconML
  - Work through weak instrument scenarios
  - Calculate LATE

- **Reflection (15 min):**
  - How does LATE differ from ATE?
  - When can IV identification fail?

#### Day 4: Difference-in-Differences - Part 1
- **Theory (45 min):**
  - Read: Causal Inference What If - DiD chapter
  - Understand: Parallel trends assumption
  - Learn: When DiD is applicable

- **Practice (60 min):**
  - Implement basic DiD estimator
  - Visualize parallel trends
  - Use panel data example

- **Reflection (15 min):**
  - How do you test parallel trends?
  - What violates parallel trends assumption?

#### Day 5: Difference-in-Differences - Part 2
- **Theory (45 min):**
  - Read: Recent DiD developments (staggered adoption, heterogeneous effects)
  - Understand: Issues with TWFE in staggered settings
  - Learn: Modern DiD estimators

- **Practice (60 min):**
  - Implement DiD with multiple time periods
  - Try event study designs
  - Use DoWhy or custom implementation

- **Reflection (15 min):**
  - Why has DiD research evolved recently?
  - When is classic DiD not enough?

#### Day 6: Week 5 Review & Application Project
- **Review (30 min):**
  - Review advanced methods

- **Mini-Project (90 min):**
  - **Project:** Panel data causal analysis
  - Find panel dataset with policy change
  - Implement DiD estimation
  - Check parallel trends
  - Perform robustness checks
  - Create publication-quality plots

---

### Week 6: Heterogeneous Treatment Effects

#### Day 1: Introduction to Treatment Heterogeneity
- **Theory (45 min):**
  - Read: EconML documentation on CATE
  - Understand: Why heterogeneity matters
  - Learn: Individual vs. average effects

- **Practice (60 min):**
  - Explore heterogeneous effects in simulated data
  - Visualize treatment effects across subgroups
  - Understand conditional average treatment effects

- **Reflection (15 min):**
  - Why is CATE important for personalization?
  - When might average effects be misleading?

#### Day 2: Meta-Learners (S, T, X-learner)
- **Theory (45 min):**
  - Read: CausalML documentation on meta-learners
  - Understand: Different meta-learner approaches
  - Learn: When to use each

- **Practice (60 min):**
  - Implement S-learner, T-learner, X-learner using CausalML
  - Compare performance on benchmark data
  - Visualize CATE estimates

- **Reflection (15 min):**
  - What are trade-offs between meta-learners?
  - Which works best in different scenarios?

#### Day 3: Causal Forests
- **Theory (45 min):**
  - Read: EconML documentation on causal forests
  - Understand: How causal trees differ from prediction trees
  - Learn: Honest splitting

- **Practice (60 min):**
  - Implement causal forests using EconML
  - Compare with meta-learners
  - Interpret forest-based CATE estimates

- **Reflection (15 min):**
  - Why are causal forests powerful?
  - What are computational considerations?

#### Day 4: Double Machine Learning (DML)
- **Theory (45 min):**
  - Read: EconML documentation on DML
  - Understand: Debiased machine learning
  - Learn: Orthogonalization and cross-fitting

- **Practice (60 min):**
  - Implement DML using EconML
  - Use different ML models (random forests, gradient boosting)
  - Compare with traditional methods

- **Reflection (15 min):**
  - Why is debiasing necessary?
  - How does DML handle high-dimensional confounders?

#### Day 5: Uplift Modeling
- **Theory (45 min):**
  - Read: CausalML book chapters on uplift
  - Understand: Targeting based on treatment effects
  - Learn: Uplift curves and evaluation

- **Practice (60 min):**
  - Implement uplift models using CausalML
  - Evaluate using uplift curves
  - Apply to marketing-style dataset

- **Reflection (15 min):**
  - How is uplift modeling used in industry?
  - What makes uplift evaluation challenging?

#### Day 6: Week 6 Review & Phase 2 Project
- **Review (30 min):**
  - Consolidate all estimation methods

- **Phase 2 Project (90 min):**
  - **Project:** Complete heterogeneous treatment effect analysis
  - Use dataset with rich covariates
  - Estimate CATE using multiple methods
  - Compare method performance
  - Identify most responsive subgroups
  - Create policy recommendations
  - Professional notebook with visualizations

---

## Phase 3: Causal Discovery & Advanced Topics (Weeks 7-9)
**Goal:** Learn how to discover causal structure and explore advanced applications

### Week 7: Causal Discovery

#### Day 1: Introduction to Causal Discovery
- **Theory (45 min):**
  - Read: causal-learn documentation and tutorials
  - Understand: Can we learn causal structure from data?
  - Learn: Key assumptions (causal sufficiency, faithfulness)

- **Practice (60 min):**
  - Install causal-learn and gcastle
  - Run simple discovery examples
  - Visualize learned graphs

- **Reflection (15 min):**
  - What makes causal discovery hard?
  - When is structure learning useful?

#### Day 2: Constraint-Based Methods (PC Algorithm)
- **Theory (45 min):**
  - Read: Papers/tutorials on PC algorithm
  - Understand: Conditional independence testing
  - Learn: How PC builds graphs incrementally

- **Practice (60 min):**
  - Implement PC algorithm using causal-learn
  - Test on known ground-truth graphs
  - Understand equivalence classes (CPDAGs)

- **Reflection (15 min):**
  - Why can't PC always identify unique DAG?
  - How does sample size affect discovery?

#### Day 3: Score-Based Methods
- **Theory (45 min):**
  - Read: Tutorials on GES, NOTEARS
  - Understand: Scoring functions and search
  - Learn: Continuous optimization for DAGs

- **Practice (60 min):**
  - Implement score-based discovery using gcastle
  - Compare PC with score-based methods
  - Try NOTEARS algorithm

- **Reflection (15 min):**
  - What are advantages of score-based vs. constraint-based?
  - How do you choose between methods?

#### Day 4: Functional Causal Models (LiNGAM)
- **Theory (45 min):**
  - Read: LiNGAM documentation and papers
  - Understand: How non-Gaussianity enables identification
  - Learn: When can we identify unique DAG

- **Practice (60 min):**
  - Implement LiNGAM using causal-learn
  - Test on linear non-Gaussian data
  - Compare with other discovery methods

- **Reflection (15 min):**
  - Why does non-Gaussianity help?
  - What happens with Gaussian noise?

#### Day 5: Time Series Causal Discovery
- **Theory (45 min):**
  - Read: Time series causality literature
  - Understand: Granger causality vs. true causality
  - Learn: Temporal ordering constraints

- **Practice (60 min):**
  - Apply causal discovery to time series data
  - Use PCMCI or similar algorithms
  - Visualize temporal graphs

- **Reflection (15 min):**
  - How does time help with causal discovery?
  - What unique challenges arise in time series?

#### Day 6: Week 7 Review & Discovery Project
- **Review (30 min):**
  - Compare different discovery algorithms

- **Mini-Project (90 min):**
  - **Project:** Causal structure learning
  - Choose dataset without known causal graph
  - Apply multiple discovery algorithms
  - Compare learned structures
  - Validate using domain knowledge
  - Discuss uncertainty and limitations

---

### Week 8: Sensitivity Analysis and Robustness

#### Day 1: Introduction to Sensitivity Analysis
- **Theory (45 min):**
  - Read: DoWhy refutation documentation
  - Understand: Why sensitivity analysis matters
  - Learn: Types of sensitivity tests

- **Practice (60 min):**
  - Use DoWhy's refutation methods
  - Test sensitivity to unobserved confounding
  - Run placebo tests

- **Reflection (15 min):**
  - How sensitive are typical analyses?
  - What makes results robust?

#### Day 2: Refutation Methods
- **Theory (45 min):**
  - Read: DoWhy paper on refutation
  - Understand: Different refutation strategies
  - Learn: Random common cause, subset refutation

- **Practice (60 min):**
  - Implement all DoWhy refutation methods
  - Apply to previous analyses
  - Interpret refutation results

- **Reflection (15 min):**
  - Which refutations are most informative?
  - How do you report sensitivity?

#### Day 3: Bounds and Partial Identification
- **Theory (45 min):**
  - Read: Literature on partial identification
  - Understand: Bounds under violations
  - Learn: When point identification isn't possible

- **Practice (60 min):**
  - Calculate bounds on treatment effects
  - Implement sensitivity analysis formulas
  - Visualize how bounds change with assumptions

- **Reflection (15 min):**
  - Are bounds useful in practice?
  - How wide are bounds typically?

#### Day 4: Falsification Tests
- **Theory (45 min):**
  - Read: Literature on falsification
  - Understand: Testing assumptions
  - Learn: Negative controls, placebo tests

- **Practice (60 min):**
  - Implement falsification tests
  - Use negative outcome controls
  - Test parallel trends in DiD

- **Reflection (15 min):**
  - What assumptions can be tested?
  - What assumptions are untestable?

#### Day 5: Multiple Testing and Specification Curves
- **Theory (45 min):**
  - Read: Specification curve analysis papers
  - Understand: Researcher degrees of freedom
  - Learn: How to show robustness across specifications

- **Practice (60 min):**
  - Create specification curves
  - Test multiple model specifications
  - Visualize robustness

- **Reflection (15 min):**
  - How do you avoid p-hacking?
  - What makes evidence convincing?

#### Day 6: Week 8 Review & Robustness Project
- **Review (30 min):**
  - Review all robustness techniques

- **Mini-Project (90 min):**
  - **Project:** Comprehensive sensitivity analysis
  - Take previous causal analysis
  - Apply all sensitivity methods
  - Create robustness dashboard
  - Write up findings with uncertainty quantification

---

### Week 9: Advanced Applications

#### Day 1: Causal Inference with Text Data
- **Theory (45 min):**
  - Read: Papers on text-based causal inference
  - Understand: Using text as treatment or confounder
  - Learn: Text embeddings for causal analysis

- **Practice (60 min):**
  - Work with text data in causal framework
  - Use embeddings as confounders
  - Estimate treatment effects with text

- **Reflection (15 min):**
  - How does text add complexity?
  - What assumptions are needed?

#### Day 2: Causal Inference with Images
- **Theory (45 min):**
  - Read: Computer vision + causality papers
  - Understand: Images in causal graphs
  - Learn: Representation learning for causality

- **Practice (60 min):**
  - Use image features in causal analysis
  - Extract representations from pre-trained models
  - Estimate effects with image data

- **Reflection (15 min):**
  - When are images useful for causal inference?
  - What unique challenges arise?

#### Day 3: Fairness and Causal Inference
- **Theory (45 min):**
  - Read: Papers on counterfactual fairness
  - Understand: Path-specific effects
  - Learn: Fair machine learning through causality

- **Practice (60 min):**
  - Implement counterfactual fairness
  - Decompose effects through different paths
  - Apply to COMPAS or similar dataset

- **Reflection (15 min):**
  - How does causality help with fairness?
  - What are limitations?

#### Day 4: Causal Reinforcement Learning - Part 1
- **Theory (45 min):**
  - Read: Introduction to causal RL
  - Understand: How causality improves RL
  - Learn: Causal models of environments

- **Practice (60 min):**
  - Work through causal RL tutorials
  - Implement simple causal RL example
  - Compare with standard RL

- **Reflection (15 min):**
  - Why is causality important for RL?
  - What problems does it solve?

#### Day 5: Causal Reinforcement Learning - Part 2
- **Theory (45 min):**
  - Read: Advanced causal RL topics
  - Understand: Counterfactual reasoning in RL
  - Learn: Applications to safe AI

- **Practice (60 min):**
  - Implement counterfactual planning
  - Work with environments with known causal structure
  - Explore recent research implementations

- **Reflection (15 min):**
  - How mature is causal RL?
  - What are open problems?

#### Day 6: Week 9 Review & Phase 3 Project
- **Review (30 min):**
  - Review advanced topics

- **Phase 3 Project (90 min):**
  - **Project:** Choose advanced application
  - Options: text-based causal analysis, fairness study, or causal RL
  - Implement complete pipeline
  - Compare with non-causal baseline
  - Document unique challenges
  - Present findings

---

## Phase 4: Real-World Applications & Mastery (Weeks 10-12)
**Goal:** Apply causal inference to real problems and develop portfolio projects

### Week 10: Industry Applications Deep Dive

#### Day 1: Marketing and A/B Testing
- **Theory (45 min):**
  - Read: Case studies on uplift modeling in marketing
  - Understand: Beyond average treatment effects
  - Learn: Targeting and personalization

- **Practice (60 min):**
  - Analyze marketing campaign data
  - Implement uplift-based targeting
  - Calculate ROI improvements

- **Reflection (15 min):**
  - How does causal thinking improve marketing?
  - What are business implications?

#### Day 2: Healthcare and Medicine
- **Theory (45 min):**
  - Read: Medical causal inference case studies
  - Understand: Treatment effect heterogeneity in medicine
  - Learn: Personalized medicine approaches

- **Practice (60 min):**
  - Analyze clinical dataset
  - Estimate heterogeneous treatment effects
  - Identify patient subgroups

- **Reflection (15 min):**
  - What makes medical causal inference challenging?
  - How do you handle safety considerations?

#### Day 3: Economics and Policy Evaluation
- **Theory (45 min):**
  - Read: Policy evaluation case studies
  - Understand: Program evaluation frameworks
  - Learn: Cost-benefit analysis with causal effects

- **Practice (60 min):**
  - Analyze policy intervention data
  - Estimate policy impact
  - Make recommendations

- **Reflection (15 min):**
  - How does causal inference inform policy?
  - What are ethical considerations?

#### Day 4: Tech and Product Analytics
- **Theory (45 min):**
  - Read: Tech company case studies (Uber, Netflix, etc.)
  - Understand: Product experimentation at scale
  - Learn: Network effects and interference

- **Practice (60 min):**
  - Analyze product feature impact
  - Handle interference in social networks
  - Estimate network effects

- **Reflection (15 min):**
  - What unique challenges arise in tech?
  - How do you scale causal inference?

#### Day 5: Finance and Econometrics
- **Theory (45 min):**
  - Read: Financial econometrics applications
  - Understand: Event studies, market impacts
  - Learn: Synthetic control for financial data

- **Practice (60 min):**
  - Analyze financial event impact
  - Use synthetic control methods
  - Handle time-series complications

- **Reflection (15 min):**
  - How does causal inference apply to finance?
  - What assumptions are most critical?

#### Day 6: Week 10 Review & Application Comparison
- **Review (30 min):**
  - Compare methods across domains

- **Exercise (90 min):**
  - **Task:** Create cross-domain comparison document
  - Summarize methods used in each industry
  - Identify common patterns
  - Note domain-specific challenges
  - Create reference guide for future use

---

### Week 11: Portfolio Project 1

#### Days 1-5: Major Project Development
**Choose one significant project (2 hours/day × 5 days = 10 hours total)**

**Project Options:**

**Option A: End-to-End Marketing Campaign Analysis**
- Objective: Optimize marketing spend using causal inference
- Dataset: Kaggle marketing dataset or simulate realistic scenario
- Tasks:
  - Estimate treatment effects of different marketing channels
  - Identify customer segments with highest uplift
  - Build targeting model
  - Calculate ROI improvements
  - Create business presentation
  - Write technical report

**Option B: Healthcare Treatment Personalization**
- Objective: Identify which patients benefit most from treatment
- Dataset: Medical dataset (MIMIC-III, or public clinical trial)
- Tasks:
  - Estimate heterogeneous treatment effects
  - Identify patient subgroups
  - Validate using clinical knowledge
  - Address ethical considerations
  - Create visualization dashboard
  - Write clinical-style report

**Option C: Policy Impact Evaluation**
- Objective: Assess impact of policy intervention
- Dataset: Find policy change with panel data
- Tasks:
  - Apply DiD or synthetic control
  - Test assumptions rigorously
  - Perform sensitivity analysis
  - Create policy brief
  - Visualize results for non-technical audience
  - Write policy recommendation

**Option D: Causal Discovery in Domain of Interest**
- Objective: Learn causal structure from observational data
- Dataset: Choose from your field
- Tasks:
  - Apply multiple discovery algorithms
  - Validate with domain expertise
  - Perform sensitivity analysis
  - Compare with literature
  - Create interactive visualization
  - Write research paper style report

**Project Requirements:**
- Complete analysis pipeline
- Rigorous assumption checking
- Sensitivity analysis
- Professional visualizations
- Technical documentation
- Executive summary
- Code on GitHub

#### Day 6: Project Presentation Preparation
- **Task (2 hours):**
  - Prepare presentation slides
  - Practice explaining to technical and non-technical audiences
  - Refine visualizations
  - Prepare FAQ/defense of choices

---

### Week 12: Portfolio Project 2 & Course Completion

#### Days 1-4: Second Major Project
**Choose a different domain/method from Week 11 (2 hours/day × 4 days = 8 hours)**

Follow same structure as Week 11 but focus on different application or method to demonstrate breadth.

#### Day 5: Course Review and Reflection
- **Activities (2 hours):**
  - Review all phases and concepts
  - Identify areas for continued learning
  - Update personal causal inference roadmap
  - Organize all notes and projects
  - Create personal reference guide
  - Write reflection on learning journey

#### Day 6: Final Integration and Next Steps
- **Activities (2 hours):**
  - Create portfolio website/GitHub showcase
  - Write blog post on learnings
  - Identify next advanced topics to study
  - Join causal inference community
  - Plan how to apply learnings to work
  - Set up system for staying current

---

## Supplementary Resources by Phase

### Phase 1 Resources
- **Primary:** Brady Neal's course and textbook
- **Videos:** YouTube lectures on causality basics
- **Practice:** DoWhy tutorials and examples
- **Community:** PyWhy discussions on GitHub

### Phase 2 Resources
- **Primary:** CausalML book and EconML documentation
- **Videos:** Stanford GSB causal inference course
- **Practice:** Benchmark datasets (Lalonde, IHDP)
- **Papers:** Classic papers on each method

### Phase 3 Resources
- **Primary:** causal-learn documentation, research papers
- **Videos:** Recent conference tutorials (UAI, ICML)
- **Practice:** Synthetic data with known ground truth
- **Advanced:** Latest arXiv papers

### Phase 4 Resources
- **Primary:** Industry case studies and blog posts
- **Datasets:** Kaggle, UCI, domain-specific repositories
- **Inspiration:** Tech company engineering blogs
- **Community:** Conferences, Twitter/LinkedIn discussions

---

## Progress Tracking

### Weekly Checklist
- [ ] Completed all 5-6 sessions
- [ ] Took notes on key concepts
- [ ] Completed coding exercises
- [ ] Finished mini-project/review
- [ ] Identified questions/challenges
- [ ] Updated learning log

### Monthly Assessment (End of each month)
**Self-Assessment Questions:**
1. Can I explain key concepts to others?
2. Can I implement methods from scratch?
3. Have I completed phase projects?
4. What concepts need more practice?
5. How confident am I with practical applications?

**Knowledge Check:**
- After Phase 1: Can you draw DAGs and explain confounding?
- After Phase 2: Can you estimate treatment effects multiple ways?
- After Phase 3: Can you discover causal structure?
- After Phase 4: Can you complete real-world analysis?

---

## Tips for Success

### Daily Habits
1. **Consistency over intensity:** 2 hours daily beats 14 hours once per week
2. **Active learning:** Code along, don't just read
3. **Take notes:** Write concepts in your own words
4. **Ask questions:** Keep a running list
5. **Review previous days:** Spend 5 min reviewing yesterday's work

### When You Get Stuck
1. Review the fundamentals
2. Work through simpler examples
3. Consult multiple resources
4. Ask in online communities (Stack Overflow, Reddit, GitHub)
5. Take a break and return fresh
6. Don't skip - understand before moving on

### Maintaining Motivation
1. Connect to your interests: Relate examples to your domain
2. Track progress: Keep learning log
3. Share progress: Blog, Twitter, LinkedIn
4. Join community: Engage with others learning
5. Celebrate milestones: Acknowledge completion of phases
6. Apply early: Look for opportunities to use at work

### Best Practices
1. **Version control:** Keep all code in Git
2. **Documentation:** Comment code and write markdown docs
3. **Reproducibility:** Use virtual environments, seed random states
4. **Organization:** Clear folder structure for projects
5. **Visualization:** Create plots to understand concepts
6. **Testing:** Validate on synthetic data with known answers

---

## Adapting This Plan

### If You Have More Time (3-4 hours/day)
- Dive deeper into mathematical foundations
- Read original research papers
- Contribute to open-source libraries
- Write blog posts explaining concepts
- Work on additional projects

### If You Have Less Time (1 hour/day)
- Extend the plan to 24 weeks
- Focus on theory OR practice each day (alternate)
- Simplify projects
- Skip some advanced topics initially
- Return to advanced content later

### If You Have Specific Focus
**Academia/Research:**
- Emphasize theory and proofs
- Read more papers
- Focus on causal discovery
- Learn mathematical details

**Industry/Applied:**
- Emphasize implementation and interpretation
- Focus on practical libraries
- More business case studies
- Portfolio projects with real data

**Specific Domain:**
- Find domain-specific examples for each concept
- Connect with domain experts
- Read domain-specific causal inference literature
- Build projects relevant to your field

---

## Post-Course Continuous Learning

### Staying Current
1. **Follow researchers:** Twitter, Google Scholar alerts
2. **Read new papers:** arXiv cs.AI, stat.ML
3. **Attend conferences:** UAI, ICML, NeurIPS (virtual options)
4. **Join workshops:** Causal inference workshops
5. **Contribute to OSS:** PyWhy ecosystem
6. **Blog/teach:** Best way to solidify knowledge

### Advanced Topics to Explore After Completion
1. Mediation analysis and path-specific effects
2. Interference and spillover effects
3. Time-varying treatments and marginal structural models
4. Survival analysis and causal inference
5. Causal inference with missing data
6. Bayesian causal inference
7. Causal representation learning
8. Integration with deep learning

### Community Engagement
1. **GitHub:** Star and contribute to causal inference libraries
2. **Reddit:** r/CausalInference for discussions
3. **Stack Overflow:** Ask and answer questions
4. **LinkedIn/Twitter:** Share insights and connect with practitioners
5. **Local meetups:** Find or start causal inference study groups
6. **Online seminars:** Many universities host virtual causal inference seminars

---

## Appendix: Quick Reference

### Essential Python Libraries Installation
```bash
# Core causal inference
pip install dowhy econml causalml causalnex

# Causal discovery
pip install causal-learn gcastle

# Supporting libraries
pip install numpy pandas matplotlib seaborn scikit-learn
pip install networkx statsmodels scipy
pip install jupyter notebook

# Optional but useful
pip install plotly shap lime
```

### Recommended Folder Structure
```
causal-ai-learning/
├── week-01/
│   ├── day-01-theory-notes.md
│   ├── day-01-practice.ipynb
│   ├── day-06-mini-project.ipynb
│   └── ...
├── week-02/
├── ...
├── projects/
│   ├── phase1-project/
│   ├── phase2-project/
│   └── final-projects/
├── datasets/
├── papers/
└── README.md (personal learning log)
```

### Key Terminology Quick Reference

| Term | Meaning |
|------|---------|
| **ATE** | Average Treatment Effect |
| **CATE** | Conditional Average Treatment Effect |
| **DAG** | Directed Acyclic Graph |
| **SCM** | Structural Causal Model |
| **do-operator** | Intervention notation P(Y\|do(X)) |
| **Confounder** | Variable affecting both treatment and outcome |
| **Collider** | Variable affected by two others |
| **Mediator** | Variable on causal path |
| **SUTVA** | Stable Unit Treatment Value Assumption |
| **Propensity Score** | Probability of receiving treatment |
| **IPW** | Inverse Probability Weighting |
| **AIPW** | Augmented IPW (doubly robust) |
| **IV** | Instrumental Variable |
| **DiD** | Difference-in-Differences |
| **RDD** | Regression Discontinuity Design |
| **ITE** | Individual Treatment Effect |

### Useful Cheatsheets to Create
1. DAG interpretation guide
2. Method selection flowchart
3. Assumption checklist by method
4. Library comparison table
5. Common pitfalls and solutions

---

## Your Learning Journey Starts Now!

Remember: Causal inference is both an art and a science. The theory provides the foundation, but practical experience builds intuition. Don't aim for perfection - aim for consistent progress.

**Start with Day 1, and build from there. Good luck!**

---

**Last Updated:** November 2024

**Note:** This plan is designed to be flexible. Adjust pacing based on your background, learning style, and goals. The key is consistent, deliberate practice over time.
