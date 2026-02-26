# Insider Trading Network Analysis

This repository contains code and data for analyzing insider trading networks using network science methods. The analysis identifies potential coordinated trading patterns among corporate insiders by constructing networks where edges represent statistically significant co-trading relationships.

## Repository Structure

```
.
├── All trades (use for replication)/
│   └── all_trades.parquet                    # Consolidated trade data
├── Data/
│   ├── 2014q1_form345/ ... 2024q4_form345/ # Raw SEC Form 3/4/5 filings (quarterly)
│   └── Preprocessed Data (trades by year)/  # Preprocessed trade data by year
├── Data-cleaning.ipynb                       # Jupyter notebook for data preprocessing
├── Network_analysis_and_visualizations.ipynb # Main analysis and visualization notebook
├── network_cpp/
│   ├── assignment_based_distinct_weeks.cpp   # Assignment-based edge detection
│   └── match_making_distinct_weeks.cpp       # Match-making edge detection
├── null_models/
│   ├── calibrated_null_model.py              # Calibrated null model generator
│   └── insider_shuffle_null.py               # Insider shuffle null model generator
├── network_results/                          # Output directory for analysis results
└── visualizations/
    ├── eigenvector_distribution.py           # Visualization scripts
    ├── figure1_forensic_fingerprint.py
    └── figure2_rich_club_null_envelope_v2.py
```

## Data Sources

The raw data consists of SEC Form 3/4/5 filings (insider transaction reports) downloaded from the [SEC Markets Data](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets). The data is organized by quarter (2014Q1 through 2024Q4) in the `Data/` folder.

## Workflow

### 1. Data Cleaning (`Data-cleaning.ipynb`)

This notebook processes the raw SEC filings:
- Merges transaction data with reporting owner information and issuer details
- Filters out institutional/corporate entities (LLCs, LPs, INCs)
- Performs group-based imputation of missing titles
- Aggregates trades by day at the individual and firm level
- Exports cleaned trade data as Parquet files

### 2. Edge Detection (`network_cpp/`)

Two C++ scripts that generate a list of edges capturing the level of coordination:

- **`match_making_distinct_weeks.cpp`**: Creates edges using the match-making algorithms (see paper for mathematical breakdown)
- **`assignment_based_distinct_weeks.cpp`**: Creates edges based on assignment-based matching criteria (see supplementary material for mathematical breakdown)

Both scripts require compilation before use:
```bash
cd network_cpp
g++ -std=c++11 -O2 assignment_based_distinct_weeks.cpp -o assignment_based_distinct_weeks
g++ -std=c++11 -O2 match_making_distinct_weeks.cpp -o match_making_distinct_weeks
```

### 3. Network Analysis (`Network_analysis_and_visualizations.ipynb`)

The main analysis notebook performs:
- Network visualization and component analysis
- Eigenvector centrality calculations
- Egonet analysis for high-centrality nodes
- Network-based anomaly detection using the OddBall algorithm
- Statistical validation of algorithm assumptions for weighted networks

### 4. Null Models (`null_models/`)

Two null model generators test the statistical significance of observed network patterns:

- **`calibrated_null_model.py`**: Generates synthetic trade histories that preserve first-order participation statistics (trade counts, tenure lengths, buy probabilities) while randomizing timing and identities. This model:
  - Randomly permutes insiders and firms to anonymize identities
  - Preserves each insider's number of firm relationships and each firm's number of insiders
  - Maintains exact trade counts and tenure lengths per insider-firm pair
  - Randomly relocates tenure windows within the study period
  - Samples trade dates uniformly within relocated tenures
  - Uses empirical buy/sell probabilities

- **`insider_shuffle_null.py`**: Randomizes insider identities within firm/time bins while preserving actual trading timestamps. This model:
  - Partitions trades by firm and calendar bin (monthly by default)
  - Randomly permutes insider IDs within each (firm, bin) group
  - Preserves firm, date, and action (buy/sell) information
  - Tests whether original insider pairs co-traded more often than chance given the actual trading calendar

Usage:
```bash
python null_models/calibrated_null_model.py --seed 42 --out-dir network_results/calibrated_null
python null_models/insider_shuffle_null.py --seed 42 --out-dir network_results/insider_shuffle_null
```

### 5. Visualizations (`visualizations/`)

Python scripts for generating publication-quality figures:
- `figure1_forensic_fingerprint.py`: Network visualization showing coordinated trading patterns
- `figure2_rich_club_null_envelope_v2.py`: Rich club analysis with null model envelopes
- `eigenvector_distribution.py`: Distribution of eigenvector centrality scores

## Output

All analysis results are saved to `network_results/`, including:
- Edge lists from network construction
- Centrality rankings
- Anomaly scores
- Null model simulation results
- Power law fit parameters

## Dependencies

- **Python 3.x** with:
  - pandas
  - networkx
  - matplotlib
  - seaborn
  - numpy
  - scipy
- **C++ compiler** with C++11 support (g++, clang++, etc.)
- **Jupyter Notebook**

## Getting Started

1. **Clean the data**: Run `Data-cleaning.ipynb` to process raw SEC filings
   - *Note: Cleaned data is available in `Data/Trades by day (use for replication)/trades_by_day.csv`. This step can be skipped and the workflow can proceed directly to step 2.*
2. **Compile C++ scripts**: Compile the edge detection scripts in `network_cpp/`
3. **Run edge detection**: Execute the compiled C++ programs to generate network edges
4. **Analyze networks**: Run `Network_analysis_and_visualizations.ipynb` for network analysis
5. **Generate null models**: Run null model scripts to test statistical significance
6. **Create visualizations**: Execute visualization scripts in `visualizations/`

## Citation

If this code or data is used, please cite the associated publication:

```bibtex
@article{jaeger2025needles,
  title={Needles in a haystack: using forensic network science to uncover insider trading},
  author={Jaeger, Gian and Yeung, Wang Ngai and Lambiotte, Renaud},
  journal={arXiv preprint arXiv:2512.18918},
  year={2025}
}
```

## Important Notes

**Null Model Generation**: The null model scripts are currently configured to generate 10 versions of the shuffled data by default. To generate more null model datasets (e.g., for more robust statistical analysis), the number of runs in the workflow needs to be adjusted. 

**Runtime and Storage Estimates** (on a laptop with 8GB+ RAM):
- Each null model dataset takes approximately 30-60 seconds to generate
- Each dataset requires approximately 40-45MB of storage
- Generating 10³ null model datasets will take approximately 8-17 hours and will generate roughly 40-45GB of data
- Runtime scales linearly with the number of datasets; storage requirements scale proportionally

## License

[Specify your license here]

