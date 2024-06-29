<a name="readme-top"></a>

# Monte-Carlo generation of synthetic multiple sequence alignments along phylogenetic trees using a protein language model

**Lab Immersion at EPFL**  
**Lab:** Bitbol Lab – Laboratory of Computational Biology and Theoretical Biophysics  
**Professor:** Anne-Florence Bitbol  
**Supervisors**: Umberto Lupo, Damiano Sgarbossa, Cyril Antoine Malbranke


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- DESCRIPTION -->
## Description
This project generates a phylogenetic tree from a natural multiple sequence alignment (MSA) using either FastTree or IQTree. From this tree, it produces a synthetic MSA through a Metropolis–Hastings algorithm for Markov Chain Monte Carlo (MCMC) employing the probabilities given by the ESM2 model. The aim is to acquire synthetic data to fine-tune the MSA transformer.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With
* ![Python](https://img.shields.io/badge/python-3776AB?logo=python&logoColor=ffdd65&style=for-the-badge&logoWidth=)
* ![Pandas](https://img.shields.io/badge/pandas-3776AB?logo=pandas&style=for-the-badge&logoWidth=)
* ![Numpy](https://img.shields.io/badge/numpy-3776AB?logo=numpy&style=for-the-badge&logoWidth=)
* ![Torch](https://img.shields.io/badge/torch-3776AB?logo=pytorch&style=for-the-badge&logoWidth=)
* ![Bio](https://img.shields.io/badge/biopython-3776AB?logo=biopython&style=for-the-badge&logoWidth=)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* Python 3.11
* FastTree
* IQTree
* MAFFT
* HMMER

To install MAFFT:
   ```sh
   conda install -c bioconda mafft
   ```

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/Bitbol-Lab/Phylogeny-ESM2.git
   ```
2. Install the requirements
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Usage

1. You can create a synthetic MSA for a single alignment using:
   ```sh
   python main.py -f <natural_msa_path>
   ```
   You can see all available command line arguments with:
      ```sh
      python main.py -h
      ```

2. For creating a synthetic MSA for multiple alignments:
   ```sh
   python run.py
   ```
   There might be the need to change the extension of the MSA input files used by this script.
   To do so change `run.py:51`:
   ```python
   if f.endswith('.fasta'):
   ```
   You can see all available command line arguments with: 
      ```sh
      python run.py -h
      ```
#### Results
1. For obtaining hamming distances correlation results:
   ```sh
   python results.py -f <msa_natural_dir> -m <method> -o <msa_synthetic_dir>
   ```
   
   You can see all available command line arguments with:
   ```sh
   python results.py -h
   ```

2. For obtaining HMMER scores and their violin plots:
   ```sh
   python hmmer_scores.py -m <method> -s <hmm_profile_dir> -o <msa_dir> -r <output_dir>
   python violin_plot.py <synthetic_scores_dir> <natural_scores_dir>
   ```
   
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License
 Apache License 2.0

<p align="right">(<a href="#readme-top">back to top</a>)</p>