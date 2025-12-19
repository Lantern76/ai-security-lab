#!/bin/bash

# AI Security Lab - Repository Setup Script
# Run this in your project directory to create the folder structure

echo "Creating AI Security Lab folder structure..."

# Create week directories
mkdir -p week-01-python-foundations
mkdir -p week-02-math-and-data
mkdir -p week-03-ml-algorithms
mkdir -p week-04-neural-networks
mkdir -p week-05-deep-learning
mkdir -p week-06-production-ml
mkdir -p week-07-adversarial-ml
mkdir -p week-08-adversarial-ml-practice
mkdir -p week-09-llm-security
mkdir -p week-10-capstone
mkdir -p week-11-portfolio
mkdir -p week-12-integration

# Create additional directories
mkdir -p sample_data
mkdir -p docs
mkdir -p utils

# Create placeholder README files in each week folder
echo "# Week 1: Python Foundations

## Topics Covered
- Variables, functions, classes
- File I/O and JSON
- Data structures
- Security event logging

## Projects
- Password Validator
- Security Event Logger
- Threat Detection System
" > week-01-python-foundations/README.md

echo "# Week 2: Math and Data

## Topics Covered
- Vectors and similarity
- Matrices and batch operations
- Linear transformations
- Pandas data manipulation
- Statistics for ML

## Projects
- Threat Similarity Detector
- Batch Threat Scorer
- Feature Transformer
" > week-02-math-and-data/README.md

echo "# Week 3: ML Algorithms

## Topics Covered
- What is learning?
- Loss functions
- Gradient descent
- Classical ML algorithms

## Projects
*Coming soon*
" > week-03-ml-algorithms/README.md

# Create empty README placeholders for future weeks
for i in 04 05 06 07 08 09 10 11 12; do
    echo "# Week $i

*Content coming soon*
" > week-$i-*/README.md 2>/dev/null
done

echo "✓ Folder structure created!"
echo ""
echo "Next steps:"
echo "1. Copy your existing code into week-01 and week-02 folders"
echo "2. git init"
echo "3. git add ."
echo "4. git commit -m 'Initial setup: AI Security Lab'"
echo "5. Create repo on GitHub"
echo "6. git remote add origin https://github.com/YOUR_USERNAME/ai-security-lab.git"
echo "7. git push -u origin main"
