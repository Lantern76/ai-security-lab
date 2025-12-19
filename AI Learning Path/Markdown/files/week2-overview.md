# Week 2: Mathematical Foundations for ML

## Overview

**Duration:** 7 days  
**Theme:** Math as notation for ideas you already understand  
**End Goal:** ML-ready data pipeline with mathematical intuition

## Learning Objectives

By the end of this week, you will:
- Understand vectors and matrices in ML context
- Use NumPy as mathematical notation
- Manipulate data with Pandas fluently
- Apply statistical thinking to security problems
- Build complete preprocessing pipelines

## Daily Breakdown

| Day | Topic | Key Concept | Project |
|-----|-------|-------------|---------|
| 1 | Vectors | Data as points in space | Threat similarity detector |
| 2 | Matrices | Batch operations | Batch threat scorer |
| 3 | Linear Transforms | What matrix multiply means | Feature transformer |
| 4 | Pandas | Labeled data manipulation | Security log analyzer |
| 5 | Statistics | Distributions and anomalies | Statistical anomaly detector |
| 6 | Preprocessing | Scaling, encoding, cleaning | Preprocessing pipeline |
| 7 | Integration | Complete pipeline | ML-ready security system |

## Prerequisites

- Week 1 complete (Python fundamentals)
- NumPy installed
- Pandas installed
- Scikit-learn installed

## Conceptual Framework

### The ML Data Flow

```
Raw Data → Feature Engineering → Preprocessing → ML Model → Predictions
    ↑              ↑                  ↑              ↑
   Day 4         Day 3           Day 5-6        Week 3+
```

This week builds everything up to the model.

### Mathematical Intuition

| Math Concept | Security Meaning | ML Application |
|--------------|------------------|----------------|
| Vector | One data point | Input to model |
| Matrix | Many data points | Batch processing |
| Dot product | Similarity | Distance, predictions |
| Linear transform | Feature combination | Neural network layers |
| Mean/Std | Normal behavior | Anomaly baselines |
| Correlation | Feature relationships | Feature selection |

## Success Criteria

### Technical Skills
- [ ] Can perform vector and matrix operations
- [ ] Can load, filter, group data with Pandas
- [ ] Can calculate and interpret statistics
- [ ] Can scale, encode, and impute data
- [ ] Built working preprocessing pipeline

### Conceptual Understanding
- [ ] Can explain why scaling matters
- [ ] Can explain what matrix multiplication does
- [ ] Can explain z-scores in security context
- [ ] Can explain one-hot vs label encoding
- [ ] Can connect math to ML algorithms

## Files to Create

```
week2/
├── day1_vectors.py
├── day2_matrices.py
├── day3_transforms.py
├── day4_pandas.py
├── day5_statistics.py
├── day6_preprocessing.py
├── day7_pipeline/
│   ├── generate_security_data.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── anomaly_detection.py
│   └── security_pipeline.py
└── security_log.csv
```

## Resources

- NumPy documentation: https://numpy.org/doc/
- Pandas documentation: https://pandas.pydata.org/docs/
- Scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html

## Connection to Career Goals

These skills directly map to job requirements:
- "Experience with data preprocessing" → Day 6
- "Statistical analysis" → Day 5
- "Feature engineering" → Day 3, Day 7
- "Data manipulation with Pandas" → Day 4

## Next Week Preview

**Week 3: Machine Learning Theory**

With your data pipeline complete, you'll learn:
- What IS learning? (Mathematical formulation)
- Loss functions and optimization
- Classical ML algorithms from scratch
- Model evaluation and selection

The foundation is set. Now we train models.
