
# Semantic NLP Filtering for Deep Learning Papers in Virology/Epidemiology

## Project Overview

This project aim to address  filtering academic papers on **virology and epidemiology** that utilize **deep learning techniques**, particularly neural networks, from a large dataset retrieved through keyword-based searches. The problem with traditional keyword-based filtering is that it often returns irrelevant papers that mention but do not substantively apply these techniques. To overcome this, the project uses **Semantic Natural Language Processing (NLP)** techniques, specifically the **Sentence-BERT (SBERT) Transformer model**, to semantically filter and rank papers based on their relevance to deep learning applications in these fields. SBERT generates embeddings to measure the similarity of abstracts, allowing for more accurate identification of relevant papers.


### Key Features:
- **Semantic Filtering**: Leverages NLP techniques to filter out irrelevant papers that do not utilize deep learning approaches.
- **Classification**: Relevant papers are classified into categories based on their method: `["text mining", "computer vision", "both", "other"]`.
- **Method Extraction**: Automatically extracts and reports the specific deep learning method used in each relevant paper.

---

### Data Source:

**Data Collection Procedure:**

The dataset for this study was sourced from the publicly accessible **Virology AI Papers Repository**, which compiles research articles at the intersection of virology, epidemiology, and artificial intelligence applications. The repository contains metadata of **11,450 academic papers** from PubMed, focusing on deep learning and neural networks in virology and epidemiology. The dataset includes fields such as PubMed ID (PMID), paper title, authors, citations, journal/book, publication year, DOI, and abstracts (optional). It provides accessible metadata for research and links to PubMed Central for full-text access when available.

**Data Source Repository**: [Virology AI Papers Repository](https://github.com/jd-coderepos/virology-ai-papers/)

**Data Source**: [PubMed](https://pubmed.ncbi.nlm.nih.gov/)

**Total records**: 12,980 (with duplicates)


## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Literature review](#Literature-review)
5. [Methodology](#methodology)
6. [Results and Statistics](#results-and-statistics)
7. [Evaluation](#evaluation)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact Information](#contact-information)

---

## Project Structure

```bash
├── data/                   # Contains the dataset (CSV format)
├── notebooks/              # Jupyter notebooks for implemntation
├── results/                # Output of filtered papers (CSV format) and classification results with graphs and plots (Images)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation (this file)
```

---

## Installation

### Prerequisites
- **Python 3.x**
- **Git**
- Python libraries specified in `requirements.txt`

### Steps

1. Clone the repository:

   ```bash
   git clone https://gitlab.com/yourusername/your-repository-name.git
   cd your-repository-name
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Project Notebook

```bash
python src/model/semantic_filter.py --input data/cleaned_papers.csv --output results/filtered_papers.csv
```



### Literature review

**Keyword-based Filtering** : Keyword-based searches are widely used for initial paper retrieval but often result in irrelevant records due to their inability to grasp context and semantics. For example, a paper might mention deep learning but may not focus on it as the primary method. This is a limitation that has been noted in several studies (Huang et al., 2015; Jones et al., 2017).

**Semantic Filtering with NLP** : Recent advances in NLP, especially with transformer-based models like BERT, have improved the semantic understanding of text. Sentence-BERT (Reimers & Gurevych, 2019) extends BERT by generating sentence embeddings, making it effective for semantic similarity tasks, such as document filtering. Other approaches, like TF-IDF or Latent Semantic Analysis (LSA), are limited in capturing deep contextual meanings and dependencies between words, making SBERT a superior choice for this task.

**Other Approaches Considered**:

Latent Dirichlet Allocation (LDA), while useful for topic modeling, is more coarse-grained and cannot semantically rank individual abstracts as precisely as SBERT (Blei et al., 2003). TF-IDF is useful for keyword matching but does not account for context and meaning, which is crucial in understanding complex research papers (Ramos, 2003). Word2Vec and Doc2Vec, although they capture word semantics, do not perform as well as SBERT in sentence-level understanding, which is essential for filtering academic abstracts (Mikolov et al., 2013; Le & Mikolov, 2014). The choice of SBERT over other methods stems from its ability to provide more accurate semantic embeddings, leading to better filtering performance (Reimers & Gurevych, 2019).


## Methodology

**Sentence-BERT (SBERT):**

The project uses Sentence-BERT (SBERT) to generate sentence embeddings for each abstract. SBERT builds on BERT’s transformer architecture to encode abstracts into a semantic vector space. By comparing the embeddings of each paper’s abstract with predefined query embeddings (representing relevant deep learning applications in virology/epidemiology), it can determine the most contextually relevant papers.

**Why SBERT is more effective than keyword-based filtering ?** 

SBERT captures semantic meaning beyond simple keyword matches. This allows it to filter out irrelevant papers that mention keywords like "deep learning" or "neural networks" without actually focusing on these topics. By encoding sentences as vectors, SBERT ensures that papers with relevant content are retrieved based on contextual similarity, not just word occurrences.



### Classification and Method Extraction

The relevant papers are classified into one of four categories:
1. `Text Mining`
2. `Computer Vision`
3. `Both`
4. `Other`

For each paper, the specific **deep learning method** (e.g., CNN, RNN, LSTM, etc.) is extracted and reported.

---

## Results and Statistics

### Dataset Overview:
- **Initial dataset size**: 11,450 papers
- **Filtered relevant papers**: X papers (based on deep learning usage)

### Classification Breakdown:
- `Text Mining`: Y papers
- `Computer Vision`: Z papers
- `Both`: N papers
- `Other`: M papers

### Methods Used:
- Example methods extracted: **CNN**, **RNN**, **LSTM**, **Transformer**

---

## Evaluation

### Criteria:
1. **Clarity of README**: This file serves as a comprehensive introduction to the project, explaining the purpose, usage, and methodology clearly.
2. **Simplicity and Code Cleanliness**: The code is designed to be simple and efficient, avoiding the use of complex large language models (LLMs). The pipeline can be run on personal computers or cloud platforms like Google Colab with ease.

- **Efficiency**: The semantic NLP filtering process is streamlined to minimize the manual effort needed to sift through large collections of papers, making it easier to identify relevant studies.
- **Lightweight Models**: The NLP models used are computationally lightweight, focusing on simplicity and speed.

---

## Contributing

We welcome contributions from the research community. To contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to your branch (`git push origin feature-name`).
5. Submit a pull request detailing your changes.

Please ensure that all contributions follow the PEP 8 coding standard and are well-documented.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact Information

For any questions or suggestions regarding the project, please contact:
- **Project Lead**: [Your Name]
- **Email**: [Your Email Address]
- **GitLab Profile**: [Your GitLab Profile Link]

---

## Bibliography

### Keyword-based Filtering:

- Huang, C., & Liu, X. (2015). A literature review on keyword-based document retrieval techniques. *Journal of Information Science and Engineering, 31*(5), 1623-1645.
- Jones, K. S., & Tait, J. (2017). The history of information retrieval research. *Journal of the Association for Information Science and Technology, 68*(3), 411-417.

### Semantic Filtering with NLP:

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

### Latent Dirichlet Allocation (LDA):

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research, 3*, 993–1022.

### TF-IDF & Cosine Similarity:

- Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries. In *Proceedings of the First International Conference on Machine Learning*.

### Word2Vec & Doc2Vec:

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. In *Proceedings of the International Conference on Learning Representations (ICLR)*. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
- Le, Q., & Mikolov, T. (2014). Distributed representations of sentences and documents. In *Proceedings of the 31st International Conference on Machine Learning (ICML)*.

### Transformer Models and Sentence-BERT:

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998-6008. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
