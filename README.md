## README

### Investigating the role of in-situ user expectations in Web search

This code provides a structured approach to perform statistical analyses on data related to user activities, sessions, and their associated metrics.

## Overview

The code is organized into various sections, including data loading, preprocessing, statistical testing, and result analysis. It's primarily structured to handle data from CSV files, preprocess it, perform various statistical tests, and provide consolidated results.

#### Requirements:

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- Scipy
- sklearn

#### Usage:

Ensure the data files (`df_task.csv`, `df_query_act.csv`, `df_user.csv`) are placed in the appropriate directory.

1. Run the main script:

```bash
python statistical_test.py
```

2. The script will load the data, preprocess it, and perform the statistical tests.

3. The results will be saved in specified dataframes like `df_p`, `df_p_task`, and `df_p_position`.

#### Functions Overview:

- `load_and_preprocess_data()`: Loads CSV data and performs initial preprocessing.
- `add_score()`: Utility function to append statistical scores to a dataframe.
- `dync_exp()`: Processes session-based data.
- `perform_statistical_tests()`: Handles the initial set of statistical tests.
- `perform_additional_statistical_tests()`: Handles additional statistical tests, including "previous experience" and "in-situ continuity and feedback".
- `main_analysis()`: Main function that serves as the entry point for the analysis.

#### Datasets:

The dataset used in this work is from a user study, which will be released after the end of the project.

#### Publication:

If you use this code, please cite our publication:

```latex
@article{WANG2023103300,
title = {Investigating the role of in-situ user expectations in Web search},
journal = {Information Processing & Management},
volume = {60},
number = {3},
pages = {103300},
year = {2023},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2023.103300},
url = {https://www.sciencedirect.com/science/article/pii/S0306457323000377},
author = {Ben Wang and Jiqun Liu},
keywords = {User search expectation, Interactive information retrieval, Online searching, User satisfaction}
}
```



## Acknowledgment

This work is supported by the National Science Foundation (NSF), USA grant IIS-2106152 and a grant from the Seed Funding Program of the Data Institute for Societal Challenges, the University of Oklahoma.



For any issues or questions regarding the code, please [contact us](mailto:benw@ou.edu). 
