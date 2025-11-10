# Causal AI Learning Plan: 2 Hours Daily

_A Practical, Structured 12-Week Journey from Basics to Mastery_

## 📋 Quick Overview

**Duration:** 12 weeks (84 days)  
**Daily Commitment:** 2 hours  
**Total Time:** ~168 hours  
**Schedule:** 6 days/week + 1 review day

### ⏰ Your Daily 2-Hour Block

```
┌─────────────────────────────────────┐
│ 0:00-0:15  Warm-up & Review         │
│ 0:15-0:45  Theory & Concepts        │
│ 0:45-1:45  Hands-on Practice        │
│ 1:45-2:00  Reflection & Planning    │
└─────────────────────────────────────┘
```

---

## 🎯 Pre-Learning Setup

### Technical Requirements

```bash
# Create your learning environment
conda create -n causal-ai python=3.9
conda activate causal-ai

# Core installations (Week 0)
pip install notebook jupyterlab
pip install numpy pandas matplotlib seaborn
pip install scikit-learn statsmodels

# Causal libraries (install as needed)
pip install dowhy  # Week 1
pip install econml  # Week 5
pip install causalml  # Week 6
pip install causal-learn  # Week 8
```

### Workspace Organization

```
causal-ai-journey/
├── 📚 resources/
│   ├── books/
│   ├── papers/
│   └── cheatsheets/
├── 📝 notes/
│   ├── daily-logs/
│   ├── concepts/
│   └── questions.md
├── 💻 code/
│   ├── week-01/
│   ├── week-02/
│   └── ...
├── 🚀 projects/
│   ├── mini-projects/
│   └── portfolio/
└── 📊 progress/
    ├── tracker.xlsx
    └── reflections.md
```

### Learning Tools Setup

1. **Note-taking:** Obsidian or Notion for concept maps
2. **Code:** VSCode with Python & Jupyter extensions
3. **Version Control:** Git repository for all work
4. **Time Tracking:** Toggl or simple spreadsheet
5. **Community:** Join PyWhy Discord & r/CausalInference

---

## 📅 Phase 1: Foundations (Weeks 1-3)

_Building Causal Intuition_

### 🗓️ Week 1: From Correlation to Causation

#### **Day 1 (Monday): The Causality Mindset**

**🎯 Learning Objectives:**

- Distinguish correlation from causation
- Identify real-world confounding examples
- Understand Simpson's Paradox

**📚 Theory (30 min):**

1. Read: Brady Neal Ch. 1 Introduction (pages 1-10)
2. Watch: [Judea Pearl - The Book of Why Talk](https://www.youtube.com/watch?v=ZaPV1OSEpHw) (first 15 min)

**💻 Practice (60 min):**

```python
# Exercise 1: Simpson's Paradox Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Berkeley admissions data
# URL: https://raw.githubusercontent.com/wadefagen/datasets/master/berkeley-admissions/data.csv
data = pd.read_csv('berkeley_admissions.csv')

# Task 1: Show overall admission rates by gender
# Task 2: Show department-specific rates
# Task 3: Explain the paradox

# Your code here...
```

**📊 Mini-Challenge:**
Find 3 examples of correlation ≠ causation in news headlines today

**✍️ Reflection Questions:**

1. Why is causal inference hard?
2. What confounders exist in my domain?
3. How would I explain Simpson's Paradox to a colleague?

**📝 Deliverable:**
Create a notebook demonstrating Simpson's Paradox with visualizations

---

#### **Day 2 (Tuesday): Introduction to DAGs**

**🎯 Learning Objectives:**

- Draw causal diagrams
- Identify causal paths
- Use NetworkX for DAG visualization

**📚 Theory (30 min):**

1. Read: Pearl's "Causal Inference in Statistics" - Chapter 1
2. Study: DAG notation and terminology

**💻 Practice (60 min):**

```python
# Exercise 2: Your First DAGs
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import gcm

# Create DAGs for these scenarios:
# 1. Ice cream sales → Swimming pool drownings (with temperature as confounder)
# 2. Education → Income (with ability as confounder)
# 3. Smoking → Lung cancer (with genetics as confounder)

def create_and_visualize_dag(edges, title):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True,
            node_color='lightblue',
            node_size=1500,
            font_size=10,
            arrows=True)
    plt.title(title)
    return G

# Example:
edges = [('Temperature', 'Ice_Cream_Sales'),
         ('Temperature', 'Drownings')]
dag1 = create_and_visualize_dag(edges, 'Ice Cream and Drownings')
```

**📊 Mini-Challenge:**
Draw DAG for: "Does working from home increase productivity?"

**✍️ Reflection:**

- What makes a good causal diagram?
- How do I know if my DAG is complete?

---

#### **Day 3 (Wednesday): Confounders, Mediators, and Colliders**

**🎯 Learning Objectives:**

- Identify three basic DAG patterns
- Understand when to control for variables
- Recognize collider bias

**📚 Theory (30 min):**

1. Read: Brady Neal Ch. 2 - "Graphical Causal Models"
2. Watch: "Confounders, Mediators, and Colliders" video

**💻 Practice (60 min):**

```python
# Exercise 3: Identifying Causal Patterns
import numpy as np
import pandas as pd

# Generate data for each pattern
np.random.seed(42)
n = 1000

# Pattern 1: Confounder
# Z → X, Z → Y, X → Y
Z = np.random.normal(0, 1, n)  # Confounder
X = 2 * Z + np.random.normal(0, 1, n)  # Treatment
Y = 3 * X + 2 * Z + np.random.normal(0, 1, n)  # Outcome

df_confound = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

# Task: Show spurious correlation without controlling for Z
# Then show true effect after controlling

# Pattern 2: Mediator
# X → M → Y
X = np.random.normal(0, 1, n)
M = 2 * X + np.random.normal(0, 1, n)  # Mediator
Y = 3 * M + np.random.normal(0, 1, n)

# Task: Show total effect vs direct effect

# Pattern 3: Collider
# X → Z ← Y
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
Z = X + Y + np.random.normal(0, 0.1, n)  # Collider

# Task: Show no correlation between X and Y
# Then show spurious correlation when conditioning on Z
```

**📊 Mini-Challenge:**
Find example of collider bias in real research (hint: survivorship bias)

---

#### **Day 4 (Thursday): The Backdoor Criterion**

**🎯 Learning Objectives:**

- Apply backdoor criterion
- Find adjustment sets
- Use DoWhy for identification

**📚 Theory (30 min):**

1. Read: Brady Neal Ch. 2 - "Backdoor Criterion"
2. Understand: Blocking backdoor paths

**💻 Practice (60 min):**

```python
# Exercise 4: Finding Adjustment Sets
from dowhy import CausalModel

# Complex DAG scenario
causal_graph = """
    digraph {
        U1[label="Unobserved"];
        U2[label="Unobserved"];

        Age -> Income;
        Age -> Health;
        Education -> Income;
        Exercise -> Health;
        Income -> Health;
        Genetics -> Health;
        Genetics -> Exercise;
        U1 -> Income;
        U1 -> Health;
        U2 -> Exercise;
        U2 -> Health;
    }
"""

# Task 1: Find all backdoor paths from Exercise to Health
# Task 2: Identify minimal adjustment set
# Task 3: Use DoWhy to verify

model = CausalModel(
    data=your_data,
    treatment='Exercise',
    outcome='Health',
    graph=causal_graph
)

# Identify effect
identified = model.identify_effect()
print(identified)
```

---

#### **Day 5 (Friday): d-Separation and Independence**

**🎯 Learning Objectives:**

- Master d-separation rules
- Test conditional independence
- Connect graphs to probability

**📚 Theory (30 min):**

1. Read: "d-separation without tears"
2. Practice: d-separation exercises

**💻 Practice (60 min):**

```python
# Exercise 5: d-Separation Testing
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils

# Create test data
def test_dseparation(dag, data, X, Y, Z_set):
    """
    Test if X and Y are d-separated given Z_set
    """
    # Statistical independence test
    p_value = fisherz(data, X, Y, Z_set)

    # Graph-based d-separation check
    is_dsep = dag.is_dseparated(X, Y, Z_set)

    return {
        'statistical_independent': p_value > 0.05,
        'graphically_dseparated': is_dsep,
        'p_value': p_value
    }

# Test cases:
# 1. Chain: A → B → C (A ⊥ C | B?)
# 2. Fork: A ← B → C (A ⊥ C | B?)
# 3. Collider: A → B ← C (A ⊥ C | B?)
```

---

#### **Day 6 (Saturday): Week 1 Review & Integration**

**🎯 Objectives:**

- Consolidate week's learning
- Complete mini-project
- Plan Week 2

**📚 Review (30 min):**

- Revisit key concepts
- Review your notes
- List questions

**💻 Mini-Project (90 min):**

**Project: "Causal Analysis of Employee Satisfaction"**

Dataset: HR Analytics (simulated or Kaggle)

Tasks:

1. Draw complete causal diagram
2. Identify all confounders for: Training → Performance
3. Show Simpson's Paradox in the data
4. Test d-separation assumptions
5. Find valid adjustment sets
6. Create presentation notebook

**Deliverable Structure:**

```markdown
# Employee Satisfaction Causal Analysis

## 1. Business Problem

[Define the causal question]

## 2. Causal Model

[Draw and justify your DAG]

## 3. Confounding Analysis

[Identify and explain confounders]

## 4. Statistical Tests

[Show correlations vs causal effects]

## 5. Recommendations

[What interventions would you suggest?]
```

---

### 🗓️ Week 2: Causal Models and Interventions

#### **Day 7 (Monday): Structural Causal Models**

**🎯 Learning Objectives:**

- Define SCMs mathematically
- Simulate data from SCMs
- Understand structural equations

**📚 Theory (30 min):**

1. Read: Brady Neal Ch. 3 - "The Flow of Association and Causation"
2. Study: SCM = {Variables, Equations, Noise}

**💻 Practice (60 min):**

```python
# Exercise 6: Building Your First SCM
class StructuralCausalModel:
    def __init__(self, equations, noise_models):
        self.equations = equations
        self.noise_models = noise_models

    def generate_data(self, n_samples=1000):
        # Generate noise
        noise = {var: model(n_samples)
                for var, model in self.noise_models.items()}

        # Compute variables using structural equations
        data = {}
        # Topological ordering important!

        return pd.DataFrame(data)

    def intervene(self, do_dict):
        # Modify equations based on intervention
        pass

# Example: Job Market SCM
def create_job_market_scm():
    equations = {
        'ability': lambda n: n['u_ability'],
        'education': lambda n: 0.7 * data['ability'] + n['u_education'],
        'job_quality': lambda n: 0.5 * data['education'] + 0.3 * data['ability'] + n['u_job'],
        'income': lambda n: 1000 * data['job_quality'] + 500 * data['education'] + n['u_income']
    }

    noise_models = {
        'u_ability': lambda n: np.random.normal(0, 1, n),
        'u_education': lambda n: np.random.normal(0, 0.5, n),
        'u_job': lambda n: np.random.normal(0, 0.3, n),
        'u_income': lambda n: np.random.normal(0, 200, n)
    }

    return StructuralCausalModel(equations, noise_models)
```

---

#### **Day 8 (Tuesday): Interventions and do-operator**

**🎯 Learning Objectives:**

- Distinguish P(Y|X) from P(Y|do(X))
- Perform graph surgery
- Simulate interventions

**📚 Theory (30 min):**

1. Read: "The do-operator explained"
2. Understand: Why conditioning ≠ intervening

**💻 Practice (60 min):**

```python
# Exercise 7: Interventions vs Observations
import dowhy.gcm as gcm

# Build causal model
causal_model = gcm.StructuralCausalModel(graph)

# Observational: P(Y|X=x)
observational = data[data['X'] == x]['Y'].mean()

# Interventional: P(Y|do(X=x))
# Method 1: Graph surgery
def graph_surgery(scm, intervention):
    modified_scm = scm.copy()
    # Remove incoming edges to intervened variables
    # Set variable to intervention value
    return modified_scm

# Method 2: Using DoWhy
samples = gcm.interventional_samples(
    causal_model,
    interventions={'X': lambda: x},
    num_samples=1000
)
interventional = samples['Y'].mean()

print(f"P(Y|X={x}): {observational}")
print(f"P(Y|do(X={x})): {interventional}")
print(f"Confounding bias: {observational - interventional}")
```

---

#### **Day 9 (Wednesday): Introduction to do-calculus**

**🎯 Learning Objectives:**

- Understand three rules of do-calculus
- Apply rules to simple graphs
- Check identifiability

**📚 Theory (30 min):**

1. Read: Pearl's do-calculus rules
2. Study: When causal effects are identifiable

**💻 Practice (60 min):**

```python
# Exercise 8: Applying do-calculus Rules
from dowhy.causal_identifier import CausalIdentifier

# Rule 1: Insertion/deletion of observations
# P(Y|do(X), Z, W) = P(Y|do(X), W) if Y ⊥ Z | X, W in G_X̅

# Rule 2: Action/observation exchange
# P(Y|do(X), do(Z), W) = P(Y|do(X), Z, W) if Y ⊥ Z | X, W in G_X̅Z̲

# Rule 3: Insertion/deletion of actions
# P(Y|do(X), do(Z), W) = P(Y|do(X), W) if Y ⊥ Z | X, W in G_X̅Z(W)

def check_identifiability(graph, treatment, outcome):
    """
    Check if causal effect is identifiable
    """
    model = CausalModel(graph=graph)
    identifier = CausalIdentifier(model, treatment, outcome)

    try:
        estimand = identifier.identify_effect()
        return True, estimand
    except:
        return False, None

# Test on various graphs
graphs = {
    'confounded': "X←Z→Y, X→Y",
    'instrumental': "Z→X→Y, U→X, U→Y",
    'frontdoor': "X→M→Y, U→X, U→Y"
}

for name, graph in graphs.items():
    identifiable, estimand = check_identifiability(graph, 'X', 'Y')
    print(f"{name}: Identifiable = {identifiable}")
    if identifiable:
        print(f"Estimand: {estimand}")
```

---

#### **Day 10 (Thursday): Front-door Criterion**

**🎯 Learning Objectives:**

- Apply front-door adjustment
- Understand when it's needed
- Compare with backdoor

**📚 Theory (30 min):**

1. Read: Front-door criterion explanation
2. Study: Classic smoking → cancer example

**💻 Practice (60 min):**

```python
# Exercise 9: Front-door Adjustment Implementation
def frontdoor_adjustment(data, X, M, Y):
    """
    Implement front-door adjustment
    P(Y|do(X)) = ΣₘP(M=m|X)ΣₓP(Y|M=m,X=x)P(X=x)
    """
    # Step 1: P(M|X)
    p_m_given_x = data.groupby([X, M]).size() / data.groupby(X).size()

    # Step 2: P(Y|M,X)
    p_y_given_mx = data.groupby([M, X])[Y].mean()

    # Step 3: P(X)
    p_x = data[X].value_counts(normalize=True)

    # Combine
    effect = 0
    for m_val in data[M].unique():
        for x_val in data[X].unique():
            effect += (p_m_given_x[1, m_val] - p_m_given_x[0, m_val]) * \
                     p_y_given_mx[m_val, x_val] * p_x[x_val]

    return effect

# Simulate smoking → tar → cancer with hidden confounder
np.random.seed(42)
n = 10000

# Hidden confounder (genetics)
genetics = np.random.binomial(1, 0.3, n)

# Smoking (affected by genetics)
smoking = np.random.binomial(1, 0.2 + 0.4 * genetics, n)

# Tar deposits (mediator, only affected by smoking)
tar = np.random.binomial(1, 0.1 + 0.7 * smoking, n)

# Lung cancer (affected by tar and genetics)
cancer = np.random.binomial(1, 0.05 + 0.3 * tar + 0.2 * genetics, n)

data = pd.DataFrame({
    'smoking': smoking,
    'tar': tar,
    'cancer': cancer,
    'genetics': genetics
})

# Compare estimates
naive = data.groupby('smoking')['cancer'].mean().diff().iloc[-1]
frontdoor = frontdoor_adjustment(data, 'smoking', 'tar', 'cancer')

print(f"Naive estimate: {naive:.3f}")
print(f"Front-door estimate: {frontdoor:.3f}")
```

---

#### **Day 11 (Friday): Counterfactual Reasoning**

**🎯 Learning Objectives:**

- Compute counterfactuals from SCMs
- Understand three steps: Abduction, Action, Prediction
- Apply to real scenarios

**📚 Theory (30 min):**

1. Read: Brady Neal Ch. 4 - "Counterfactuals"
2. Study: Twin network representation

**💻 Practice (60 min):**

```python
# Exercise 10: Computing Counterfactuals
class CounterfactualSCM:
    def __init__(self, scm):
        self.scm = scm

    def compute_counterfactual(self, observed_data, intervention, target):
        """
        Three steps of counterfactual inference:
        1. Abduction: Infer noise from observed data
        2. Action: Apply intervention
        3. Prediction: Compute outcome
        """
        # Step 1: Abduction - infer noise values
        noise = self.infer_noise(observed_data)

        # Step 2: Action - modify SCM
        modified_scm = self.scm.intervene(intervention)

        # Step 3: Prediction - compute with inferred noise
        counterfactual_data = modified_scm.predict(noise)

        return counterfactual_data[target]

    def infer_noise(self, observed_data):
        # Inverse of structural equations
        pass

# Example: "What if this patient had exercised?"
patient_data = {
    'age': 45,
    'exercise': 0,  # Didn't exercise
    'diet_quality': 3,
    'health_score': 65  # Observed outcome
}

# Counterfactual query
cf_health = compute_counterfactual(
    observed_data=patient_data,
    intervention={'exercise': 1},
    target='health_score'
)

print(f"Actual health score: {patient_data['health_score']}")
print(f"Counterfactual (if exercised): {cf_health}")
print(f"Individual treatment effect: {cf_health - patient_data['health_score']}")
```

---

#### **Day 12 (Saturday): Week 2 Integration**

**💻 Week 2 Project: "Build a Complete Causal Analysis System"**

Create an end-to-end system that:

1. Takes a dataset and causal graph
2. Identifies causal effects
3. Estimates effects using multiple methods
4. Computes counterfactuals
5. Validates assumptions

```python
class CausalAnalysisSystem:
    def __init__(self, data, graph):
        self.data = data
        self.graph = graph
        self.model = CausalModel(data, graph)

    def full_analysis(self, treatment, outcome):
        results = {}

        # 1. Identification
        results['identified'] = self.identify(treatment, outcome)

        # 2. Estimation (multiple methods)
        results['estimates'] = self.estimate_all(treatment, outcome)

        # 3. Counterfactuals
        results['counterfactuals'] = self.compute_counterfactuals()

        # 4. Sensitivity analysis
        results['sensitivity'] = self.sensitivity_analysis()

        # 5. Visualization
        self.visualize_results(results)

        return results
```

---

### 🗓️ Week 3: Potential Outcomes & Causal Estimands

#### **Day 13-18: Detailed Daily Plans**

[Continuing with same detailed format for remaining days...]

**Day 13: Potential Outcomes Framework**

- Rubin Causal Model basics
- Fundamental problem of causal inference
- Connection to SCMs

**Day 14: Treatment Effects Zoo**

- ATE, ATT, ATC, LATE, CATE
- When to use each estimand
- Practical examples

**Day 15: Identification Assumptions**

- SUTVA, Ignorability, Positivity
- Testing assumptions
- Sensitivity analysis

**Day 16: Randomized Experiments**

- Gold standard causality
- Design considerations
- Analysis of experiments

**Day 17: Observational Studies**

- Challenges and solutions
- Covariate selection
- Diagnostic checks

**Day 18: Week 3 Project**

- Complete RCT analysis
- Compare to observational approach
- Document limitations

---

## 📅 Phase 2: Core Methods (Weeks 4-6)

_Mastering Essential Techniques_

### 🗓️ Week 4: Matching and Weighting

#### Daily Breakdown:

- **Day 19:** Exact and coarsened matching
- **Day 20:** Propensity score estimation
- **Day 21:** Propensity score matching
- **Day 22:** Inverse probability weighting
- **Day 23:** Doubly robust methods
- **Day 24:** Week 4 Project

#### Week 4 Hands-on Project:

**"Evaluating Job Training Program"**
Using LaLonde dataset, implement:

1. Multiple matching approaches
2. Balance diagnostics
3. Sensitivity analysis
4. Comparison of estimates

### 🗓️ Week 5: Instrumental Variables & Natural Experiments

#### Daily Breakdown:

- **Day 25:** IV intuition and assumptions
- **Day 26:** Two-stage least squares
- **Day 27:** Weak instruments
- **Day 28:** Local Average Treatment Effect
- **Day 29:** Applications and examples
- **Day 30:** Week 5 Project

### 🗓️ Week 6: Difference-in-Differences & Panel Methods

#### Daily Breakdown:

- **Day 31:** DiD basics and assumptions
- **Day 32:** Parallel trends testing
- **Day 33:** Staggered adoption
- **Day 34:** Synthetic control method
- **Day 35:** Fixed effects and panel data
- **Day 36:** Week 6 Project

---

## 📅 Phase 3: Advanced Methods (Weeks 7-9)

_Modern ML-Based Approaches_

### 🗓️ Week 7: Machine Learning for Causal Inference

#### Daily Focus Areas:

- Double/debiased machine learning
- Causal forests
- Meta-learners (S, T, X, R)
- Targeted learning
- Cross-fitting techniques
- Week project: CATE estimation

### 🗓️ Week 8: Causal Discovery

#### Daily Focus Areas:

- Constraint-based methods (PC, FCI)
- Score-based methods (GES, NOTEARS)
- Functional causal models
- Time series causal discovery
- Validation techniques
- Week project: Discover causal structure

### 🗓️ Week 9: Special Topics

#### Daily Focus Areas:

- Mediation analysis
- Time-varying treatments
- Interference and spillovers
- Missing data and causality
- Causal inference in RL
- Integration week project

---

## 📅 Phase 4: Real-World Application (Weeks 10-12)

_Portfolio Development_

### 🗓️ Week 10: Domain-Specific Applications

Choose your domain and implement:

- Healthcare: Patient outcome prediction
- Marketing: Campaign optimization
- Policy: Program evaluation
- Tech: A/B testing enhancement
- Finance: Risk assessment

### 🗓️ Week 11: Comprehensive Project

Build end-to-end causal analysis:

1. Problem formulation
2. Data collection/simulation
3. Causal model specification
4. Multiple estimation approaches
5. Validation and sensitivity
6. Actionable recommendations

### 🗓️ Week 12: Portfolio & Community

- Create GitHub portfolio
- Write blog posts
- Contribute to open source
- Present findings
- Network building
- Plan continued learning

---

## 📊 Progress Tracking System

### Daily Check-in Template

```markdown
## Day [X] - [Date]

### ✅ Completed

- [ ] Theory reading (30 min)
- [ ] Coding exercise (60 min)
- [ ] Notes/reflection (15 min)
- [ ] Question logged

### 💡 Key Insights

1.
2.
3.

### 🤔 Questions/Struggles

-

### 🎯 Tomorrow's Focus

-

### ⏱️ Time Logged: **\_** hours
```

### Weekly Review Template

```markdown
## Week [X] Review

### 🎓 Concepts Mastered

- [ ]
- [ ]
- [ ]

### 💻 Code Completed

- [ ] All exercises
- [ ] Mini-project
- [ ] GitHub commits

### 📈 Self-Assessment (1-5)

- Understanding: \_\_\_
- Implementation: \_\_\_
- Confidence: \_\_\_

### 🚀 Next Week Prep

-
```

### Phase Completion Checklist

#### Phase 1 ✓

- [ ] Can draw and interpret DAGs
- [ ] Understand confounding, mediation, collision
- [ ] Can identify causal effects
- [ ] Built first causal model
- [ ] Completed 3 mini-projects

#### Phase 2 ✓

- [ ] Implemented propensity score methods
- [ ] Applied instrumental variables
- [ ] Used difference-in-differences
- [ ] Understand assumptions and limitations
- [ ] Completed method comparison project

#### Phase 3 ✓

- [ ] Applied ML to causal inference
- [ ] Discovered causal structure from data
- [ ] Handled complex scenarios
- [ ] Built reusable code library
- [ ] Completed advanced project

#### Phase 4 ✓

- [ ] Solved real-world problem
- [ ] Created portfolio pieces
- [ ] Wrote technical blog posts
- [ ] Connected with community
- [ ] Have job-ready skills

---

## 💪 Overcoming Common Challenges

### Challenge 1: "The Math is Too Hard"

**Solutions:**

- Start with intuition, then formalize
- Use simulations to understand
- Find visual explanations
- Join study groups
- Accept gradual understanding

### Challenge 2: "I Don't Have Real Data"

**Solutions:**

- Use simulation (you control ground truth!)
- Kaggle datasets
- UCI repository
- Generate synthetic data
- Public government data

### Challenge 3: "Too Many Methods to Learn"

**Solutions:**

- Master one method deeply first
- Understand when each applies
- Build personal cheat sheet
- Focus on your domain's common methods

### Challenge 4: "Can't Find Time"

**Solutions:**

- Morning routine before work
- Lunch break theory reading
- Weekend project time
- Reduce to 1 hour but stay consistent
- Track time to find gaps

### Challenge 5: "Feeling Overwhelmed"

**Solutions:**

- Review fundamentals
- Take breaks when needed
- Celebrate small wins
- Connect with others learning
- Remember: confusion is part of learning

---

## 🛠️ Practical Resources

### Essential Bookmarks

#### Documentation

- [DoWhy Docs](https://py-why.github.io/dowhy/)
- [EconML Docs](https://econml.azurewebsites.net/)
- [CausalML Docs](https://causalml.readthedocs.io/)

#### Tutorials

- [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)
- [Mixtape Sessions](https://mixtape.scunning.com/)

#### Communities

- [PyWhy Discord](https://discord.gg/cSBGb3vsZb)
- [r/CausalInference](https://reddit.com/r/causalinference)
- [Online Causal Inference Seminar](https://sites.google.com/view/ocis/)

### Code Snippet Library

```python
# Quick-start templates for common tasks

# 1. Propensity Score Matching Template
def ps_matching_pipeline(data, treatment, outcome, covariates):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    # Estimate propensity scores
    ps_model = LogisticRegression()
    ps_model.fit(data[covariates], data[treatment])
    ps = ps_model.predict_proba(data[covariates])[:, 1]

    # Check overlap
    check_overlap(ps, data[treatment])

    # Match
    treated_ps = ps[data[treatment] == 1]
    control_ps = ps[data[treatment] == 0]

    matcher = NearestNeighbors(n_neighbors=1)
    matcher.fit(control_ps.reshape(-1, 1))
    distances, indices = matcher.kneighbors(treated_ps.reshape(-1, 1))

    # Estimate effect
    treated_outcomes = data[data[treatment] == 1][outcome]
    matched_control_outcomes = data[data[treatment] == 0].iloc[indices.flatten()][outcome]

    ate = (treated_outcomes - matched_control_outcomes).mean()
    return ate

# 2. DiD Template
def difference_in_differences(data, outcome, treatment, time, treated_group):
    import statsmodels.formula.api as smf

    # Create DiD interaction
    data['did'] = data[treated_group] * data[time]

    # Run regression
    formula = f'{outcome} ~ {treated_group} + {time} + did'
    model = smf.ols(formula, data=data).fit()

    # Extract treatment effect
    treatment_effect = model.params['did']
    confidence_interval = model.conf_int().loc['did']

    return {
        'effect': treatment_effect,
        'ci_lower': confidence_interval[0],
        'ci_upper': confidence_interval[1],
        'model': model
    }

# 3. Causal Forest Template
def causal_forest_analysis(X, Y, T, W):
    from econml.dml import CausalForestDML

    # Initialize and fit
    cf = CausalForestDML(
        model_y='auto',
        model_t='auto',
        discrete_treatment=True,
        n_estimators=100,
        min_samples_leaf=10
    )

    cf.fit(Y, T, X=X, W=W)

    # Get heterogeneous effects
    cate = cf.effect(X)
    cate_lb, cate_ub = cf.effect_interval(X)

    # Feature importance
    importance = cf.feature_importances_

    return {
        'cate': cate,
        'confidence_bounds': (cate_lb, cate_ub),
        'feature_importance': importance
    }
```

### Dataset Resources

#### Benchmark Datasets

1. **LaLonde (1986)** - Job training program
2. **IHDP** - Infant health and development
3. **ACIC 2016-2019** - Competition datasets
4. **Twins** - Twin births and mortality
5. **JOBS** - Job search intervention

#### Where to Find

- [Causal Inference Datasets](https://github.com/amit-sharma/causal-inference-datasets)
- [Vanderbilt Biostatistics Datasets](http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets)
- [Harvard Dataverse](https://dataverse.harvard.edu/)

---

## 🎓 Certification & Recognition

### Building Your Credentials

#### Online Certificates

1. **Coursera** - Penn's Causal Inference Course
2. **EdX** - Harvard's Causal Diagrams
3. **Udacity** - Causal Inference Nanodegree

#### Portfolio Pieces

1. **GitHub Repository**

   - Clean, documented code
   - Multiple methods implemented
   - Real-world applications

2. **Technical Blog Posts**

   - Method explanations
   - Case studies
   - Tutorials

3. **Kaggle Notebooks**
   - Public analyses
   - Competition entries
   - Upvoted contributions

#### Community Involvement

- Answer Stack Overflow questions
- Contribute to PyWhy ecosystem
- Present at local meetups
- Publish on arXiv

---

## 🚀 After the 12 Weeks

### Immediate Next Steps

1. **Week 13-14: Integration**

   - Review all notes
   - Refactor code library
   - Polish portfolio projects

2. **Week 15-16: Specialization**

   - Choose focus area
   - Deep dive into advanced topic
   - Read recent papers

3. **Month 4+: Application**
   - Apply at work
   - Freelance projects
   - Research collaboration
   - Open source contribution

### Long-term Learning Path

#### 6 Months

- Master one specialized area
- Publish first blog post
- Complete significant project
- Join research reading group

#### 1 Year

- Conference presentation
- Contribute to major library
- Mentor others starting
- Industry application

#### 2+ Years

- Research publication
- Package development
- Thought leadership
- Teaching/training

### Staying Current

#### Weekly Habits

- Read 1 new paper
- Code 1 new technique
- Answer 1 community question

#### Monthly Goals

- Complete mini-project
- Write blog post
- Attend virtual seminar

#### Annual Objectives

- Attend conference
- Major contribution
- Expand network

---

## 💭 Final Thoughts

### Remember These Truths

1. **Causality is Hard** - Even experts disagree. Embrace uncertainty.

2. **Theory + Practice** - Neither alone is sufficient. Balance both.

3. **Domain Knowledge Matters** - Causal inference isn't just statistics.

4. **Start Simple** - Master basics before advanced methods.

5. **Community Helps** - Don't learn in isolation.

6. **Apply Early** - Look for causal questions everywhere.

7. **Document Everything** - Your future self will thank you.

8. **Confusion is Normal** - It means you're learning.

9. **Quality > Quantity** - Deep understanding beats surface knowledge.

10. **It's a Journey** - Causal thinking changes how you see the world.

---

## 📌 Quick Start Checklist

**Right Now (10 minutes):**

- [ ] Create learning folder structure
- [ ] Install Python and Jupyter
- [ ] Join PyWhy Discord
- [ ] Bookmark this guide
- [ ] Schedule first 2-hour block

**Today:**

- [ ] Install core libraries
- [ ] Download first dataset
- [ ] Read Day 1 materials
- [ ] Write learning goals

**This Week:**

- [ ] Complete Days 1-6
- [ ] Join one community
- [ ] Find accountability partner
- [ ] Share learning publicly

---

**Your causal inference journey starts now. Block 2 hours tomorrow and begin with Day 1.**

_Remember: Every expert was once a beginner who didn't give up._

---

**Version:** 2.0  
**Last Updated:** November 2024  
**Feedback:** Welcome via GitHub issues  
**License:** CC BY-SA 4.0

---

_"Correlation does not imply causation, but it does waggle its eyebrows suggestively and gesture furtively while mouthing 'look over there'."_ - Randall Munroe, xkcd
