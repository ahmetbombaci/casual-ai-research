# Causal AI Research Guide

_A Comprehensive Resource for Learning and Implementing Causal Inference and Causal Machine Learning_

## Quick Navigation

- [🎯 Start Here](#start-here-choose-your-path)
- [📚 Core Concepts](#core-concepts-explained)
- [🛠️ Tools & Libraries](#tools-and-libraries)
- [🔬 Methods & Techniques](#causal-inference-methods)
- [💡 Real-World Applications](#real-world-applications)
- [📖 Learning Resources](#learning-resources)
- [🚀 Getting Started Guide](#getting-started-roadmap)

---

## 🎯 Start Here: Choose Your Path

### I'm a...

#### **Data Scientist/ML Engineer**

→ **Your Focus:** Integrating causality into ML pipelines

- Start with: [DoWhy Quick Start](#dowhy-quick-start)
- Essential reading: "Causal AI" by Robert Osazuwa Ness
- Key skill: Heterogeneous treatment effects with EconML
- Project idea: Uplift modeling for marketing campaigns

#### **Business Analyst**

→ **Your Focus:** Data-driven decision making with causal insights

- Start with: [Business Applications](#business-and-marketing)
- Essential tool: CausalNex for Bayesian networks
- Key skill: A/B testing with causal interpretation
- Project idea: Customer churn intervention analysis

#### **Researcher/Academic**

→ **Your Focus:** Theory and advanced methods

- Start with: [Pearl's Causal Hierarchy](#pearls-causal-hierarchy)
- Essential reading: "Causality" by Judea Pearl
- Key skill: Causal discovery algorithms
- Project idea: Novel causal discovery method comparison

#### **Healthcare Professional**

→ **Your Focus:** Treatment effects and clinical trials

- Start with: [Healthcare Applications](#healthcare-and-medicine)
- Essential method: Propensity score matching
- Key skill: Confounder adjustment in observational studies
- Project idea: Treatment heterogeneity analysis

---

## 📚 Core Concepts Explained

### The Causal Ladder: Three Levels of Understanding

#### Level 1: Association (Seeing)

**Question:** "What is?"

- Statistical relationships: P(Y|X)
- Correlation, regression, clustering
- **Example:** Customers who buy organic food tend to exercise more

#### Level 2: Intervention (Doing)

**Question:** "What if I do?"

- Causal effects: P(Y|do(X))
- Experiments, A/B tests, policy changes
- **Example:** If we mandate exercise programs, will health outcomes improve?

#### Level 3: Counterfactuals (Imagining)

**Question:** "What if I had done differently?"

- Alternative histories: P(Y_x|X', Y')
- Individual-level causation
- **Example:** Would this specific patient have recovered without the treatment?

### Pearl's Causal Hierarchy

```
┌─────────────────────────────────────┐
│   Level 3: Counterfactuals          │
│   "What if things had been different?" │
│   Requires: Structural Causal Model │
├─────────────────────────────────────┤
│   Level 2: Interventions            │
│   "What if we do X?"                │
│   Requires: Causal Graph + Data     │
├─────────────────────────────────────┤
│   Level 1: Associations             │
│   "What is?"                        │
│   Requires: Data only               │
└─────────────────────────────────────┘
```

### Directed Acyclic Graphs (DAGs): Visual Causality

#### The Building Blocks

**1. Confounders** (Common Causes)

```
    Z
   ↙ ↘
  X → Y
```

- Z affects both X and Y
- Creates spurious correlation
- **Solution:** Control for Z

**2. Mediators** (Causal Chains)

```
X → M → Y
```

- M is on the causal path
- Part of the mechanism
- **Caution:** Don't control if you want total effect

**3. Colliders** (Common Effects)

```
X → Z ← Y
```

- Z is affected by both X and Y
- Creates selection bias if controlled
- **Rule:** Don't condition on colliders

#### Practical DAG Example: Education and Income

```python
# Real-world DAG: Education → Income
import networkx as nx
import matplotlib.pyplot as plt

# Define the causal structure
edges = [
    ('Intelligence', 'Education'),
    ('Family_Wealth', 'Education'),
    ('Family_Wealth', 'Income'),
    ('Education', 'Income'),
    ('Education', 'Job_Satisfaction'),
    ('Income', 'Job_Satisfaction')
]

# What to control for when estimating Education → Income?
# Answer: Family_Wealth (confounder), NOT Job_Satisfaction (collider)
```

### Structural Causal Models (SCMs): The Mathematical Framework

An SCM consists of:

1. **Variables:** {X₁, X₂, ..., Xₙ}
2. **Structural Equations:** Xᵢ = fᵢ(PAᵢ, Uᵢ)
3. **Probability Distribution:** P(U)

#### Example SCM: Simple Economic Model

```python
# Structural equations
def income(education, ability, noise):
    return 1000 * education + 500 * ability + noise

def education(ability, wealth, noise):
    return 0.5 * ability + 0.3 * wealth + noise

# This defines causal relationships, not just correlations
```

### Potential Outcomes Framework: The Alternative Perspective

**Core Concepts:**

- **Potential Outcomes:** Y(1) = outcome if treated, Y(0) = outcome if not treated
- **Causal Effect:** τᵢ = Y(1) - Y(0) for individual i
- **Fundamental Problem:** Can't observe both Y(1) and Y(0) for same individual

**Key Estimands:**

```
ATE  = E[Y(1) - Y(0)]                    # Average Treatment Effect
ATT  = E[Y(1) - Y(0)|T=1]                # Effect on the Treated
CATE = E[Y(1) - Y(0)|X=x]                # Conditional Average Effect
```

**Identifying Assumptions:**

1. **SUTVA:** No interference between units
2. **Ignorability:** Treatment assignment independent of potential outcomes (given covariates)
3. **Overlap:** 0 < P(T=1|X) < 1 for all X

---

## 🛠️ Tools and Libraries

### Library Selection Guide

| **Use Case**          | **Recommended Tool** | **Why**                          |
| --------------------- | -------------------- | -------------------------------- |
| First causal project  | DoWhy                | User-friendly, good diagnostics  |
| Heterogeneous effects | EconML               | State-of-the-art CATE methods    |
| Marketing/uplift      | CausalML             | Specialized for uplift modeling  |
| Causal discovery      | causal-learn         | Most comprehensive algorithms    |
| Business decisions    | CausalNex            | Bayesian networks, interpretable |
| Time series           | Tigramite            | Specialized for temporal data    |
| Large-scale ML        | CausalTune           | Automated hyperparameter tuning  |

### DoWhy Quick Start

```python
# Installation
pip install dowhy

# Basic workflow
import dowhy
from dowhy import CausalModel
import pandas as pd

# 1. Load your data
data = pd.read_csv('your_data.csv')

# 2. Define causal graph
model = CausalModel(
    data=data,
    treatment='treatment_variable',
    outcome='outcome_variable',
    graph="""
        graph [directed 1 node [id="treatment" label="Treatment"]
               node [id="outcome" label="Outcome"]
               node [id="confounder" label="Confounder"]
               edge [source="confounder" target="treatment"]
               edge [source="confounder" target="outcome"]
               edge [source="treatment" target="outcome"]]
    """
)

# 3. Identify causal effect
identified_estimand = model.identify_effect()

# 4. Estimate the effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

# 5. Validate with robustness checks
refutation = model.refute_estimate(
    identified_estimand, estimate,
    method_name="random_common_cause"
)

print(f"Causal Effect: {estimate.value}")
print(f"Validation: {refutation}")
```

### EconML for Heterogeneous Effects

```python
# Installation
pip install econml

# Example: Causal Forest for CATE
from econml.dml import CausalForestDML
import numpy as np

# Setup
model = CausalForestDML(
    model_y='auto',  # Automatically selects ML model
    model_t='auto',
    discrete_treatment=True,
    n_estimators=100
)

# Fit the model
model.fit(Y, T, X=X, W=W)  # W = confounders, X = effect modifiers

# Get treatment effects
treatment_effects = model.effect(X_test)

# Confidence intervals
te_lower, te_upper = model.effect_interval(X_test)

# Feature importance for heterogeneity
importance = model.feature_importances_
```

### CausalML for Uplift Modeling

```python
# Installation
pip install causalml

# Uplift modeling example
from causalml.inference.meta import LRSRegressor
from causalml.metrics import plot_gain

# S-Learner approach
lr_s = LRSRegressor()
lr_s.fit(X=features, treatment=treatment, y=outcome)

# Get uplift scores
uplift_scores = lr_s.predict(X=features_test)

# Visualize uplift
plot_gain(outcome_test, uplift_scores, treatment_test)
```

### Causal Discovery with causal-learn

```python
# Installation
pip install causal-learn

# PC Algorithm example
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

# Run PC algorithm
cg = pc(data.values, 0.05, fisherz, True, 0, -1)

# Visualize discovered graph
cg.draw_pydot_graph()

# Get adjacency matrix
adj_matrix = cg.G.graph
```

---

## 🔬 Causal Inference Methods

### Method Selection Flowchart

```
Start: Do you have randomized data?
├─ Yes → Use simple comparison of means
└─ No → Do you have a valid instrument?
    ├─ Yes → Use Instrumental Variables
    └─ No → Can you find good controls?
        ├─ Yes → Do you have many confounders?
        │   ├─ Yes → Use ML methods (Double ML, Causal Forests)
        │   └─ No → Use regression/matching
        └─ No → Can you use time variation?
            ├─ Yes → Use Difference-in-Differences or Synthetic Control
            └─ No → Causal discovery or acknowledge limitations
```

### 1. Propensity Score Methods

**When to Use:**

- Observable confounding
- Treatment/control imbalance
- Want to mimic randomized trial

**Implementation Checklist:**

- [ ] Check overlap assumption
- [ ] Balance diagnostics pre/post matching
- [ ] Sensitivity to hidden bias
- [ ] Multiple matching specifications

**Code Template:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Step 1: Estimate propensity scores
ps_model = LogisticRegression()
ps_model.fit(confounders, treatment)
propensity_scores = ps_model.predict_proba(confounders)[:, 1]

# Step 2: Check overlap
plt.hist([propensity_scores[treatment==0],
          propensity_scores[treatment==1]],
         label=['Control', 'Treated'])
plt.legend()

# Step 3: Match or weight
# Matching
matcher = NearestNeighbors(n_neighbors=1)
matcher.fit(propensity_scores[treatment==0].reshape(-1, 1))
matches = matcher.kneighbors(propensity_scores[treatment==1].reshape(-1, 1))

# IPW
weights = treatment / propensity_scores + (1 - treatment) / (1 - propensity_scores)
```

### 2. Instrumental Variables (IV)

**Valid Instruments Must:**

1. **Relevance:** Correlate with treatment
2. **Exclusion:** Affect outcome only through treatment
3. **Independence:** Uncorrelated with unobserved confounders

**Classic Examples:**

- Distance to college → Education → Earnings
- Quarter of birth → Years of schooling → Wages
- Lottery assignment → Military service → Earnings

**Two-Stage Least Squares (2SLS):**

```python
from linearmodels import IV2SLS
import pandas as pd

# First stage: Treatment ~ Instrument + Controls
# Second stage: Outcome ~ Predicted_Treatment + Controls

model = IV2SLS(dependent=outcome,
               exog=controls,
               endog=treatment,
               instruments=instrument)
results = model.fit()
print(results.summary)
```

### 3. Difference-in-Differences (DiD)

**Key Assumption:** Parallel trends in absence of treatment

**Visual Check:**

```python
# Pre-treatment trends should be parallel
import seaborn as sns

pre_treatment_data = data[data['time'] < treatment_time]
sns.lineplot(data=pre_treatment_data,
             x='time', y='outcome',
             hue='treatment_group')
```

**Implementation:**

```python
# DiD regression
import statsmodels.formula.api as smf

# Create interaction term
data['did'] = data['treated'] * data['post_period']

# Run regression
model = smf.ols('outcome ~ treated + post_period + did', data=data)
results = model.fit()

# The coefficient on 'did' is your treatment effect
treatment_effect = results.params['did']
```

### 4. Regression Discontinuity Design (RDD)

**When to Use:**

- Treatment assigned based on threshold
- Running variable cannot be manipulated

**Implementation Steps:**

```python
from rdd import rdd

# Estimate at the cutoff
result = rdd.rdd(data=df,
                  y='outcome',
                  x='running_variable',
                  c=cutoff_value,
                  fuzzy=False)  # True if fuzzy RDD

# Visualization
rdd.plot(data=df, y='outcome', x='running_variable', c=cutoff_value)
```

### 5. Synthetic Control Method

**When to Use:**

- Few treated units
- Good pre-treatment fit possible
- Aggregate-level data

**Key Steps:**

1. Create weighted combination of control units
2. Match pre-treatment outcomes
3. Compare post-treatment

```python
from synthetic_control import SyntheticControl

# Setup
sc = SyntheticControl(
    data=panel_data,
    treatment_unit='California',
    treatment_period=1989,
    outcome_variable='cigarette_sales'
)

# Fit and predict
sc.fit()
synthetic_outcome = sc.predict()

# Treatment effect
effect = actual_outcome - synthetic_outcome
```

### 6. Double/Debiased Machine Learning

**Advantages:**

- Handles high-dimensional confounders
- Avoids overfitting bias
- Flexible ML methods

**Implementation:**

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Setup DML
dml = LinearDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestClassifier(),
    discrete_treatment=True
)

# Fit
dml.fit(Y, T, X=features, W=confounders)

# Get effects with confidence intervals
effects = dml.effect(X_test)
lb, ub = dml.effect_interval(X_test)
```

---

## 💡 Real-World Applications

### Healthcare and Medicine

#### Example: Personalized Treatment Effects

**Problem:** Which patients benefit most from a new drug?

**Approach:**

```python
# Heterogeneous treatment effects in clinical trial
from econml.metalearners import TLearner

# T-Learner for heterogeneous effects
t_learner = TLearner(models=RandomForestRegressor())
t_learner.fit(Y=patient_outcomes, T=treatment, X=patient_features)

# Individual treatment effects
ite = t_learner.effect(patient_features)

# Identify subgroups
high_benefit = patient_features[ite > threshold]
```

**Key Considerations:**

- Handle missing data carefully
- Account for non-compliance
- Consider time-varying treatments
- Validate on external cohorts

### Business and Marketing

#### Example: Customer Uplift Modeling

**Problem:** Who should receive marketing promotions?

**Approach:**

```python
# Multi-treatment uplift
from causalml.inference.meta import XLearner

# Train uplift model
xl = XLearner(models=GradientBoostingRegressor())
xl.fit(X=customer_features,
       treatment=campaign_assignment,
       y=purchase_amount)

# Rank customers by uplift
uplift_scores = xl.predict(X=new_customers)

# Target top 20% with highest uplift
top_customers = new_customers[uplift_scores > np.percentile(uplift_scores, 80)]
```

**ROI Calculation:**

```python
# Expected profit from targeting
expected_profit = (
    uplift_scores * profit_per_conversion
    - cost_per_treatment
)

# Optimal treatment threshold
treat_if_profit_positive = expected_profit > 0
```

### Policy Evaluation

#### Example: Minimum Wage Impact

**Problem:** Effect of minimum wage increase on employment

**Approach:** Synthetic Control + DiD

```python
# Combine methods for robustness
from causalimpact import CausalImpact

# Define pre and post periods
pre_period = ['2010-01-01', '2014-12-31']
post_period = ['2015-01-01', '2017-12-31']

# Run causal impact analysis
ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())

# Visualize
ci.plot()
```

### A/B Testing Enhancement

#### Beyond Simple Comparisons

**CUPED (Controlled-experiment Using Pre-Experiment Data):**

```python
# Reduce variance using pre-experiment data
def cuped_estimate(Y_post, Y_pre, treatment):
    # Compute theta (optimal weight)
    theta = np.cov(Y_post, Y_pre)[0,1] / np.var(Y_pre)

    # Adjust outcomes
    Y_adjusted = Y_post - theta * (Y_pre - np.mean(Y_pre))

    # Compute treatment effect
    effect = (Y_adjusted[treatment == 1].mean() -
              Y_adjusted[treatment == 0].mean())

    return effect
```

**Stratified Experiments:**

```python
# Improve precision with stratification
from scipy import stats

def stratified_analysis(data, strata_col):
    results = []
    for stratum in data[strata_col].unique():
        stratum_data = data[data[strata_col] == stratum]

        # Within-stratum effect
        effect = (stratum_data[stratum_data.treated == 1].outcome.mean() -
                 stratum_data[stratum_data.treated == 0].outcome.mean())

        # Weight by stratum size
        weight = len(stratum_data) / len(data)
        results.append({'stratum': stratum,
                       'effect': effect,
                       'weight': weight})

    # Weighted average
    overall_effect = sum(r['effect'] * r['weight'] for r in results)
    return overall_effect, results
```

---

## 📖 Learning Resources

### 📚 Essential Books (Ranked by Accessibility)

#### Beginner-Friendly

1. **"The Book of Why" by Judea Pearl**

   - Non-technical introduction to causal thinking
   - Great for intuition building
   - Read first for motivation

2. **"Introduction to Causal Inference" by Brady Neal**
   - Free online textbook
   - ML perspective
   - Excellent progression of concepts
   - [Free PDF](https://www.bradyneal.com/causal-inference-course)

#### Intermediate

3. **"Causal AI" by Robert Osazuwa Ness**

   - Practical implementation focus
   - Python code examples
   - Integrates with modern ML
   - [Manning Publications](https://www.manning.com/books/causal-ai)

4. **"Causal Inference: The Mixtape" by Scott Cunningham**
   - Economics perspective
   - Real-world examples
   - Code in R and Python
   - [Free Online](https://mixtape.scunning.com/)

#### Advanced

5. **"Causality" by Judea Pearl**

   - The foundational text
   - Mathematical rigor
   - Complete theoretical treatment

6. **"Elements of Causal Inference" by Peters, Janzing, and Schölkopf**
   - Machine learning perspective
   - Focus on causal discovery
   - [Free PDF](https://mitpress.mit.edu/9780262037310/)

### 🎓 Online Courses (Structured Learning)

#### Free Courses

**1. Brady Neal's Course (Best Overall Free Resource)**

- **Format:** Video lectures + textbook
- **Duration:** ~20 hours
- **Link:** [Course Website](https://www.bradyneal.com/causal-inference-course)
- **Includes:** Problem sets, solutions

**2. Harvard's Causal Diagrams Course (EdX)**

- **Focus:** DAGs and graphical models
- **Duration:** 9 weeks
- **Certificate:** Available (paid)
- **Best for:** Visual learners

**3. MIT's The Challenge of World Poverty**

- **Focus:** RCTs and development economics
- **Instructors:** Esther Duflo, Abhijit Banerjee
- **Platform:** MIT OpenCourseWare

#### Paid Courses

**4. Coursera - A Crash Course in Causality**

- **University:** University of Pennsylvania
- **Duration:** 5 weeks
- **Hands-on:** R programming exercises
- **Certificate:** Included

**5. Udacity - Causal Inference Program**

- **Duration:** 4 months
- **Project-based:** Real-world applications
- **Mentorship:** Included

### 🎥 Video Resources

#### YouTube Channels

- **Brady Neal:** Causal inference lectures
- **Online Causal Inference Seminar:** Research talks
- **Simons Institute:** Advanced topics

#### Key Talks

1. "The Seven Tools of Causal Inference" - Judea Pearl
2. "Machine Learning & Causal Inference" - Susan Athey
3. "Double Machine Learning" - Victor Chernozhukov

### 📰 Blogs and Articles

#### Must-Read Blog Posts

1. "Understanding Simpson's Paradox" - Towards Data Science
2. "Causal Inference in Online Systems" - Netflix Tech Blog
3. "Causal Inference at Uber" - Uber Engineering
4. "Pearlian DAGs vs Potential Outcomes" - Andrew Gelman

#### Active Blogs

- **Judea Pearl's Blog:** Theoretical insights
- **Andrew Gelman's Blog:** Statistical perspective
- **Microsoft Research Blog:** DoWhy updates
- **PyWhy Blog:** Ecosystem news

### 🔬 Research Papers

#### Foundational Papers

1. Pearl (1995) - "Causal Diagrams for Empirical Research"
2. Rubin (1974) - "Estimating Causal Effects of Treatments"
3. Holland (1986) - "Statistics and Causal Inference"

#### Modern Developments

1. Chernozhukov et al. (2018) - "Double/Debiased Machine Learning"
2. Wager & Athey (2018) - "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"
3. Kallus (2020) - "DeepMatch: Balancing Deep Covariate Representations"

### 👥 Communities

#### Online Communities

- **PyWhy Discord:** Active community, quick help
- **r/CausalInference:** Reddit discussions
- **Cross Validated:** Stack Exchange for statistics
- **Twitter:** #CausalInference hashtag

#### Conferences

- **UAI:** Uncertainty in Artificial Intelligence
- **ICML:** Causal inference workshops
- **NeurIPS:** Causal learning track
- **ACIC:** Atlantic Causal Inference Conference

---

## 🚀 Getting Started Roadmap

### Week 1-2: Build Intuition

- [ ] Read "The Book of Why" (popular science)
- [ ] Watch Brady Neal Lectures 1-3
- [ ] Practice drawing DAGs for familiar scenarios
- [ ] Install DoWhy, run first example

### Week 3-4: Core Theory

- [ ] Study confounding, selection bias
- [ ] Learn backdoor criterion
- [ ] Understand do-calculus basics
- [ ] Implement simulation studies

### Week 5-6: First Methods

- [ ] Master propensity scores
- [ ] Try regression adjustment
- [ ] Run A/B test analysis
- [ ] Complete first real dataset analysis

### Week 7-8: Advanced Methods

- [ ] Learn instrumental variables
- [ ] Try difference-in-differences
- [ ] Explore machine learning methods
- [ ] Start personal project

### Week 9-12: Specialization

- [ ] Choose focus area
- [ ] Deep dive into relevant methods
- [ ] Complete substantial project
- [ ] Share findings (blog/GitHub)

---

## 💡 Pro Tips

### Common Pitfalls to Avoid

1. **Controlling for Colliders**

   - Never condition on common effects
   - Check DAG before adding controls

2. **Ignoring Positivity Violations**

   - Always check overlap in propensity scores
   - Be cautious extrapolating

3. **P-hacking in Causal Analysis**

   - Pre-specify analysis plan
   - Report all specifications tried

4. **Assuming Causal Discovery = Causal Truth**

   - Discovery algorithms have assumptions
   - Validate with domain knowledge

5. **Forgetting About Interference**
   - SUTVA may not hold
   - Consider spillover effects

### Best Practices Checklist

Before Analysis:

- [ ] Draw your assumed DAG
- [ ] List all assumptions explicitly
- [ ] Plan sensitivity analyses
- [ ] Consider alternative explanations

During Analysis:

- [ ] Check covariate balance
- [ ] Validate assumptions where possible
- [ ] Try multiple methods
- [ ] Document all decisions

After Analysis:

- [ ] Interpret causally, not just statistically
- [ ] Report uncertainty honestly
- [ ] Discuss limitations
- [ ] Suggest validation studies

### Quick Decision Guide

| **Scenario**                    | **Method**                | **Why**                         |
| ------------------------------- | ------------------------- | ------------------------------- |
| RCT with non-compliance         | Instrumental Variables    | Randomization as instrument     |
| Before/after with control group | Difference-in-differences | Controls for time trends        |
| Treatment at threshold          | Regression Discontinuity  | Local randomization             |
| Many confounders                | Double ML                 | Handles high dimensions         |
| Need individual effects         | Causal Forests            | Heterogeneous treatment effects |
| Time series intervention        | Interrupted Time Series   | Accounts for autocorrelation    |
| Few treated units               | Synthetic Control         | Creates comparison unit         |

---

## 📊 Tools Comparison Matrix

| Feature                | DoWhy      | EconML     | CausalML   | causal-learn | CausalNex  |
| ---------------------- | ---------- | ---------- | ---------- | ------------ | ---------- |
| **Ease of Use**        | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐         | ⭐⭐⭐⭐   |
| **Documentation**      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐       | ⭐⭐⭐⭐   |
| **Identification**     | ✅         | ❌         | ❌         | ✅           | ✅         |
| **CATE Estimation**    | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐           | ⭐⭐       |
| **Causal Discovery**   | ❌         | ❌         | ❌         | ⭐⭐⭐⭐⭐   | ⭐⭐⭐     |
| **Validation**         | ⭐⭐⭐⭐⭐ | ⭐⭐       | ⭐⭐       | ⭐           | ⭐⭐⭐     |
| **Visualization**      | ⭐⭐⭐⭐   | ⭐⭐       | ⭐⭐⭐⭐   | ⭐⭐⭐       | ⭐⭐⭐⭐⭐ |
| **Active Development** | ✅✅       | ✅✅       | ✅         | ✅           | ✅         |

---

## 🎯 Project Ideas by Skill Level

### Beginner Projects

1. **Simpson's Paradox Explorer**

   - Dataset: UC Berkeley admissions
   - Skills: Data visualization, basic statistics
   - Deliverable: Interactive notebook

2. **A/B Test Analyzer**
   - Dataset: Any online experiment
   - Skills: Hypothesis testing, effect sizes
   - Deliverable: Automated report generator

### Intermediate Projects

3. **Propensity Score Matcher**

   - Dataset: LaLonde job training data
   - Skills: Matching algorithms, balance checks
   - Deliverable: Reusable matching pipeline

4. **Marketing Campaign Uplift Model**
   - Dataset: Retail customer data
   - Skills: Uplift modeling, targeting
   - Deliverable: Customer scoring system

### Advanced Projects

5. **Causal Discovery Benchmark**

   - Dataset: Multiple synthetic/real datasets
   - Skills: Multiple discovery algorithms
   - Deliverable: Performance comparison paper

6. **Heterogeneous Effect Estimator**
   - Dataset: Clinical trial or economic data
   - Skills: ML methods, subgroup analysis
   - Deliverable: R/Python package

---

## 🔄 Keeping Current

### Weekly Routine

- **Monday:** Check arXiv for new papers
- **Wednesday:** Read one paper/blog post
- **Friday:** Code one new technique

### Monthly Goals

- Implement one new method
- Analyze one dataset causally
- Write one blog post/notebook
- Attend one online seminar

### Annual Objectives

- Contribute to one open-source project
- Submit to one conference/workshop
- Complete one substantial project
- Build network in causal inference

---

## 📝 Quick Reference Cards

### Assumption Violations: Warning Signs

| **Assumption**            | **Warning Signs**                      | **Solutions**                          |
| ------------------------- | -------------------------------------- | -------------------------------------- |
| No unmeasured confounding | Similar groups have different outcomes | Sensitivity analysis, IV, RDD          |
| Positivity                | No overlap in propensity scores        | Trim, change estimand                  |
| SUTVA                     | Units interact/interfere               | Cluster randomization, modify estimand |
| Parallel trends           | Pre-trends diverge                     | Synthetic control, match on trends     |
| Exclusion (IV)            | Instrument affects outcome directly    | Find better instrument                 |

### Code Snippets Library

```python
# Quick templates for common tasks

# 1. Balance check
def check_balance(X, treatment):
    from scipy import stats
    balanced = []
    for col in X.columns:
        treated = X.loc[treatment == 1, col]
        control = X.loc[treatment == 0, col]
        _, p_value = stats.ttest_ind(treated, control)
        balanced.append({'variable': col,
                        'p_value': p_value,
                        'balanced': p_value > 0.1})
    return pd.DataFrame(balanced)

# 2. Overlap check
def check_overlap(propensity_scores, treatment):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(propensity_scores[treatment == 0],
             alpha=0.5, label='Control', bins=30)
    plt.hist(propensity_scores[treatment == 1],
             alpha=0.5, label='Treated', bins=30)
    plt.xlabel('Propensity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Propensity Score Overlap')
    return plt

# 3. Simple bootstrap CI
def bootstrap_ci(data, func, n_bootstrap=1000, confidence=0.95):
    bootstrap_results = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = data.sample(n=n, replace=True)
        bootstrap_results.append(func(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_results, alpha/2 * 100)
    upper = np.percentile(bootstrap_results, (1 - alpha/2) * 100)
    return lower, upper
```

---

## 🚦 Next Steps

1. **Start Small:** Pick one method and master it
2. **Apply Often:** Look for causal questions in your daily work
3. **Share Knowledge:** Blog about your learnings
4. **Build Community:** Connect with other practitioners
5. **Stay Curious:** Causality changes how you see the world

---

**Last Updated:** November 2024

**Contributing:** This is a living document. Suggestions and contributions welcome via GitHub issues.

**Disclaimer:** Causal inference requires careful thought and domain expertise. These tools help, but don't replace critical thinking about your specific problem.
