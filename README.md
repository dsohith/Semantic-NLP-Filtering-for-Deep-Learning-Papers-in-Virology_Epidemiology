
# Semantic NLP Filtering for Deep Learning Papers in Virology/Epidemiology

## Project Overview

This project aim to address  filtering research papers on **virology and epidemiology** that utilise **deep learning neural network based solution**, from a large dataset retrieved through keyword-based searches. The problem with traditional keyword-based filtering is that it often returns irrelevant papers that mention but do not substantively apply these techniques. To overcome this, the project uses **Semantic Natural Language Processing (NLP)** techniques, specifically the **Sentence-BERT (SBERT) Transformer model**, to semantically filter and rank papers based on their relevance to deep learning applications in these fields. SBERT generates embeddings to measure the similarity of abstracts, allowing for more accurate identification of relevant papers.


### Key Features:
- **Semantic Filtering**: Leverages NLP techniques to filter out irrelevant papers that do not utilise deep learning approaches.
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
3. [Literature review](#Literature-review)
4. [Methodology](#methodology)
5. [Results and Statistics](#results-and-statistics)
6. [Evaluation](#evaluation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact Information](#contact-information)

---

## Project Structure

```bash
├── data/                  # Contains the dataset (CSV format)
├── myenv                  # virtual environment and installed librabries
├── results                # Output of filtered papers (CSV format) 
   ├── Images/             # classification results with graphs and plots (Images)
  
├── Semantic_NLP_Filtering_for_Deep_Learning_Papers_in_Virology_Epidemiology.ipynb              # Jupyter notebook for implemntation
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)
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
   git clone https://gitlab.com/dsohith/Semantic-NLP-Filtering-for-Deep-Learning-Papers-in-Virology_Epidemiology.git
   cd dsohith
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Project Notebook

```bash
Semantic_NLP_Filtering_for_Deep_Learning_Papers_in_Virology_Epidemiology.ipynb
```



### Literature review

**Keyword-based Filtering** : Keyword-based searches are widely used for initial paper retrieval but often result in irrelevant records due to their inability to grasp context and semantics. For example, a paper might mention deep learning but may not focus on it as the primary method. This is a limitation that has been noted in several studies (Huang et al., 2015; Jones et al., 2017).

**Semantic Filtering with NLP** : Recent advances in NLP, especially with transformer-based models like BERT, have improved the semantic understanding of text. Sentence-BERT  extends BERT by generating sentence embeddings, making it effective for semantic similarity tasks, such as document filtering (Reimers & Gurevych, 2019). Other approaches, like TF-IDF or Latent Semantic Analysis (LSA), are limited in capturing deep contextual meanings and dependencies between words, making SBERT a superior choice for this task (Blei et al., 2003).

**Other Approaches Considered**:

Latent Dirichlet Allocation (LDA), while useful for topic modeling, is more coarse-grained and cannot semantically rank individual abstracts as precisely as SBERT (Blei et al., 2003). TF-IDF is useful for keyword matching but does not account for context and meaning, which is crucial in understanding complex research papers (Ramos, 2003). Word2Vec and Doc2Vec, although they capture word semantics, do not perform as well as SBERT in sentence-level understanding, which is essential for filtering academic abstracts (Mikolov et al., 2013; Le & Mikolov, 2014). The choice of SBERT over other methods stems from its ability to provide more accurate semantic embeddings, leading to better filtering performance (Reimers & Gurevych, 2019).


## Methodology

**Sentence-BERT (SBERT):**

The project uses Sentence-BERT (SBERT) to generate sentence embeddings for each abstract. SBERT builds on BERT’s transformer architecture to encode abstracts into a semantic vector space. By comparing the embeddings of each paper’s abstract with predefined query embeddings (representing relevant deep learning applications in virology/epidemiology), it can determine the most contextually relevant papers.

**Why SBERT is more effective than keyword-based filtering ?** 

SBERT captures semantic meaning beyond simple keyword matches. This allows it to filter out irrelevant papers that mention keywords like "deep learning" or "neural networks" without actually focusing on these topics. By encoding sentences as vectors, SBERT ensures that papers with relevant content are retrieved based on contextual similarity, not just word occurrences.


**Task 1:  Semantic natural language processing techniques to filter out papers that do not meet
the criteria of utilizing deep learning approaches in virology/epidemiology.**

The first task involved filtering research papers to identify those that utilise deep learning techniques within the domains of virology or epidemiology. To achieve this,  a semantic filtering approach using embeddings derived from the abstracts of the papers. The SentenceTransformer model (all-MiniLM-L6-v2), a lightweight transformer-based model renowned for efficiently generating high-quality sentence embeddings is used. This model converts each abstract into a vector representation within a high-dimensional space, capturing the semantic meaning of the text. Subsequently,  deep learning-related keywords  which are derived from search query of data collection process are embedded using the same transformer model. This approach enables to capture the underlying semantics of each keyword in the specific context of deep learning.

The core of  filtering mechanism relied on calculating the cosine similarity between the abstract embeddings and the keyword embeddings. Cosine similarity measures the cosine of the angle between two vectors, yielding a numerical score between -1 and 1, which reflects the degree of similarity between the vectors. A higher similarity score indicates that the abstract is more likely to be related to deep learning methodologies.

To determine which papers were relevant, a statistical thresholding method is applied. Specifically,  the mean and standard deviation of the cosine similarity scores for all abstracts are calculated. The papers were then filtered based on their Deep_Learning_similarity_percentage, which was computed as the percentage similarity between the abstract and the deep learning-related keywords. Papers whose similarity score exceeded the threshold of mean - 2*standard deviation were considered relevant and retained in the dataset. This thresholding approach allows to exclude papers with weak or irrelevant connections to deep learning, ensuring that the papers  analysed were more likely to be related to the application of deep learning in virology or epidemiology. This methodology offered an efficient and semantically informed way to filter papers, capturing the nuanced meanings within the abstracts without being restricted to exact keyword matches.


**Task 2: For the papers deemed relevant, classify them according to the type of method used: ["text
mining", "computer vision", "both", "other"].**

After identifying the relevant papers, the subsequent task was to classify them according to the method employed, specifically into one of the following categories: "text mining," "computer vision," "both," or "other." The classification process was based on comparing the semantic content of each abstract to predefined embeddings representing the methods of text mining and computer vision. To achieve this, embeddings for each method by encoding brief descriptions or keywords associated with each technique using the same SentenceTransformer model. This approach ensured that the classification was grounded in the semantic essence of the methods, rather than relying on exact keyword matches.

To classify each paper,  the cosine similarity between the paper’s abstract embedding and each of the method embeddings are calculated. This resulted in a similarity score for each method, indicating the degree of alignment between the abstract and the respective method. A higher similarity score signified a closer relationship between the abstract and the method. If both text mining and computer vision methods exhibited similarity scores above a specific threshold, the paper was classified as "both." If only one method had a high similarity score, the paper was assigned to the corresponding category, either "text mining" or "computer vision." Papers that did not show significant similarity to either method were classified as "other." This classification approach allowed for a nuanced distinction between papers, as it took into account the semantic content of the abstract, ensuring that papers employing multiple approaches or less common terminology were accurately categorised.

**Task 3: Extract and report the name of the method used for each relevant paper.**

The third task involved extracting and reporting the specific deep learning method used in each relevant paper. Here, the objective was to identify the exact method from a predefined set of techniques, such as CNN, RNN, Transformer, and other deep learning architectures. The extraction process followed a similar procedure to the classification task but with a focus on pinpointing the method that best matched the abstract. To facilitate this, embeddings for each method, using  keywords that represent each deep learning technique, as had done previously.

For each paper,  the cosine similarity between its abstract embedding and the embeddings for all predefined deep learning methods are computed. The resulting similarity scores reflected the degree of alignment between the abstract and each method. The method with the highest similarity score was identified as the most likely technique used in the paper. In cases where a clear difference in similarity scores existed, the paper was assigned the method that yielded the highest score. This process provided an effective means of identifying the specific deep learning method utilised, even in instances where the method was not explicitly stated by name. It allowed for the accurate extraction of methods based on semantic meaning, ensuring that the classification was not reliant on exact keyword matches, but rather on the actual content of the abstract.

The use of cosine similarity for identifying the deep learning method was particularly advantageous, as it accounted for the diverse and complex language often found in research papers. Rather than depending on rigid keyword matching, which may overlook subtleties or variations in terminology, the use of semantic embeddings enabled us to correctly match papers to their respective deep learning techniques, even when the methods were described differently or indirectly referenced.


## Results and Statistics

### Dataset Overview:
- **Initial dataset size**: 11,450 papers

| Column            | Non-Null Count  | Data Type |
|-------------------|-----------------|-----------|
| PMID              | 11,450          | int64     |
| Title             | 11,450          | object    |
| Authors           | 11,450          | object    |
| Citation          | 11,450          | object    |
| First Author      | 11,450          | object    |
| Journal/Book      | 11,450          | object    |
| Publication Year  | 11,450          | int64     |
| Create Date       | 11,450          | object    |
| PMCID             | 6,450           | object    |
| NIHMS ID          | 956             | object    |
| DOI               | 10,969          | object    |
| Abstract          | 11,237          | object    |



- **Distribution of Abstract Length**

| Statistic | Value       |
|-----------|-------------|
| Count     | 11,237      |
| Mean      | 1,723.08    |
| Standard Deviation (std) | 575.94    |
| Minimum   | 4           |
| 25%       | 1,386       |
| Median (50%) | 1,699     |
| 75%       | 1,983       |
| Maximum   | 20,492      |


### Data Pre processing:

- Initial size: 11450 
- Missing abstracts: 213
- Filtered size: 11237
- Converts text to lowercase
- Remove special characters
- Lemmatize words to base form
- Remove stopwords

![Alt Text](results/Images/Overview_of_abstracts.png)

### Task 1 Results: Filtering papers that do not meet the criteria of utilizing deep learning approaches

**Filtering using SBERT model**

- **Distribution of similarity percentage**

![Alt Text](results/Images/Distribution_of_similarity_percentage.png)

- Total papers with Abstract: 11237
- Total relevant papers similarity with deep learning keywords: 11040

 **Comparision with key word based search**

| Keyword Match Count | Count |
|---------------------|-------|
| 0                   | 4313  |
| 1                   | 3736  |
| 2                   | 1884  |
| 3                   | 933   |
| 4                   | 260   |
| 5                   | 66    |
| 6                   | 31    |
| 7                   | 10    |
| 8                   | 4     |
  

The comparison between the SBERT-based similarity approach and the keyword-based search for identifying relevant papers reveals key differences in effectiveness. The SBERT-based method identified 11,040 papers as relevant, demonstrating its ability to capture papers discussing deep learning in a broader, more nuanced context, even if the exact deep learning keywords were not present. This approach leverages semantic embeddings to compare the meaning of the abstracts to predefined deep learning-related keywords, ensuring a more comprehensive identification process. In contrast, the keyword-based search identified papers based solely on the presence of specific deep learning keywords. The results show a steep drop-off in the number of papers as the number of matching keywords decreases, with 4,313 papers having no keyword matches at all. This suggests that keyword-based methods tend to overlook papers that may discuss deep learning using different terms or indirect references. Thus, while the keyword-based search offers a simpler and more direct method, it is limited in its scope, whereas the SBERT-based similarity approach provides a more inclusive and context-aware classification, highlighting its superior performance in filtering relevant papers related to deep learning in virology and epidemiology.

### Task 2 Results: Classification into ["text mining", "computer vision", "both", "other"].

**Classification based on SBERT model**

| Method Type      | Count |
|------------------|-------|
| other            | 3751  |
| text mining      | 3085  |
| Both             | 2502  |
| computer vision  | 1702  |

![Alt Text](results/Images/Classification_based_on_SBERT_model.png).

**Classification based on Keyword based search**

| Category         | Count |
|------------------|-------|
| other            | 9300  |
| text mining      | 1485  |
| computer vision  | 447   |
| both             | 5     |


![Alt Text](results/Images/Classification_based_on_keyword_based_search.png).

The comparison between the SBERT-based classification and the keyword-based search classification reveals notable differences in how each method categorizes research papers. The SBERT model identifies a more diverse range of approaches, with a significant number of papers (3,751) categorized as "Other," reflecting its ability to capture a wide variety of methods beyond predefined categories such as text mining and computer vision. This is followed by a substantial number of papers classified under text mining (3,085), both text mining and computer vision (2,502), and computer vision (1,702), highlighting the model's capacity to identify multimodal approaches and capture semantic nuances. In contrast, the keyword-based search produces a larger proportion of papers classified as Other (9,300), indicating that the method's reliance on specific keywords leads to a loss of papers discussing relevant methodologies using alternative terminology. It also identifies fewer papers under text mining (1,485) and computer vision (447), suggesting that the search fails to account for variations in terminology or the broader scope of deep learning methods. Furthermore, the small number of papers classified as using both methods (only 5) emphasizes the limitations of keyword-based searches in recognizing multimodal approaches. Overall, the SBERT model provides a more comprehensive and semantically aware classification, while the keyword-based approach tends to miss relevant papers due to its reliance on exact keyword matches.


### Task 3 Results: Method used for each relevant paper.


- **Count of papers belongs to specific methods.**

| Method Used                         | Count |
|-------------------------------------|-------|
| text analytics                      | 1,195 |
| feedforward neural network           | 960   |
| multimodal neural network            | 935   |
| pretrained language model            | 756   |
| natural language processing          | 747   |
| RNN                                  | 713   |
| text mining                          | 650   |
| vision model                         | 624   |
| LSTM                                 | 498   |
| image processing                     | 448   |
| vision transformer                   | 311   |
| CNN                                  | 305   |
| sequence-to-sequence models          | 273   |
| neural net algorithm                 | 266   |
| long short-term memory network       | 233   |
| convolutional neural network         | 200   |
| textual data analysis                | 184   |
| recurrent neural network             | 183   |
| generative deep learning             | 160   |
| vision algorithms                    | 130   |
| computer graphics and vision         | 128   |
| computer vision                      | 122   |
| language modeling                    | 101   |
| generative AI                        | 100   |
| multimodal model                     | 100   |
| foundation model                     | 87    |
| NLP                                  | 78    |
| object recognition                   | 61    |
| generative artificial intelligence   | 57    |
| continuous diffusion model           | 46    |
| speech and language technology       | 45    |
| transformer models                   | 43    |
| language processing                  | 42    |
| diffusion-based generative model     | 34    |
| computational semantics              | 33    |
| generative models                    | 29    |
| self-attention models                | 26    |
| large language model                 | 24    |
| attention-based neural networks      | 19    |
| diffusion model                      | 18    |
| transformer                          | 13    |
| LLM                                  | 13    |
| transformer networks                 | 11    |
| transformer-based model              | 10    |
| generative diffusion model           |  7    |
| generative language model            |  6    |
| transformer architecture             |  6    |
| multilayer perceptron                |  5    |
| GRNN                                 |  3    |
| computational linguistics            |  2    |
            

**Top 10 methods used in papers belongs to Virology/Epidemiology


![Alt Text](results/Images/Classification_based_on_keyword_based_search.png).

---


## Conclusion

This project successfully streamlined the early stages of article collection for review by employing Semantic NLP filtering with the Sentence-BERT (SBERT) model, effectively minimizing the need for manual scanning and filtering of numerous articles. The SBERT-based approach significantly outperformed traditional keyword-based searches by capturing semantic meaning and context, leading to more accurate identification and categorization of papers focused on deep learning applications in virology and epidemiology. The project not only filtered relevant papers but also classified them into key methodological categories such as text mining, computer vision, and both, while accurately extracting the specific deep learning techniques used. This approach provides an efficient and powerful tool for researchers to navigate large datasets with precision and reduced manual effort.

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
- **Project Lead**: Sohith Dhavaleswarapu
- **Email**: sohith.dhavaleswarapu@gmail.com
- **GitLab Profile**: https://github.com/dsohith

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
