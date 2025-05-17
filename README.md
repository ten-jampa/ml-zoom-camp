# Machine Learning Zoomcamp Journey

This repository contains my learning journey, homework solutions, and projects completed as part of the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course by DataTalks.Club.

## Course Overview

Machine Learning Zoomcamp is a comprehensive, hands-on course that covers the fundamentals of machine learning and its practical applications. The course is structured into several modules, each focusing on different aspects of the ML lifecycle:

1. Introduction to Machine Learning
2. Machine Learning for Regression
3. Machine Learning for Classification
4. Evaluation Metrics for Classification
5. Deploying Machine Learning Models
6. Decision Trees and Ensemble Learning
7. Deep Learning
8. Serverless Deep Learning
9. Kubernetes and TensorFlow Serving
10. Capstone Project

## Environment Setup

To run the notebooks and code in this repository, you'll need to set up a Python environment with the required dependencies.

### Option 1: Using Conda (Recommended)

1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create a new conda environment:
```bash
conda create -n mlzoomcamp python=3.11
conda activate mlzoomcamp
```

3. Install required packages:
```bash
conda install numpy pandas scikit-learn jupyter tensorflow keras matplotlib seaborn xgboost
```

### Option 2: Using pip and venv

1. Create a virtual environment:
```bash
python -m venv mlzoomcamp-env
source mlzoomcamp-env/bin/activate  # On Linux/Mac
# or
.\mlzoomcamp-env\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Repository Structure

- `01-intro/`: Introduction to Machine Learning
- Additional folders will be added as I progress through the course

## Usage

Each module folder contains:
- Jupyter notebooks with code examples and exercises
- Homework solutions
- Additional resources and notes

To run the notebooks:
1. Activate your environment:
```bash
conda activate mlzoomcamp  # if using conda
# or
source mlzoomcamp-env/bin/activate  # if using venv
```

2. Start Jupyter:
```bash
jupyter notebook
```


## Acknowledgments

Special thanks to:
- [DataTalks.Club](https://datatalks.club/) for creating and maintaining this excellent course
- Alexey Grigorev and all the instructors for their amazing work

## Disclaimer

This repository is meant for learning and reference purposes. If you're also taking the course, make sure to attempt the assignments yourself before looking at any solutions.

## License

This project is open source and available under the [MIT License](LICENSE).
