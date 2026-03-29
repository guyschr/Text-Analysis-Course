# Analysis of Disciplinary Weight in Neanderthal Language Research

## 📋 Project Overview

This project applies **Natural Language Processing (NLP)** and **Computational Linguistics** to systematically analyze 39 seminal academic articles (1989–2025) on Neanderthal linguistic capabilities. The research quantifies shifting evidentiary "weight" across three primary disciplinary domains:
- **Anatomy** (morphological & neuroanatomical evidence)
- **Genetics** (genomic & molecular evidence)
- **Archaeology/Behavioral** (cultural & archaeological evidence)

---

## 🔍 Research Questions

1. **Disciplinary Weight**: How has the prominence and emphasis of different scientific fields evolved in the debate over Neanderthal language over the past 35 years?

2. **Thematic Structure**: Can unsupervised machine learning (LDA topic modeling) identify latent thematic clusters that align with or transcend traditional disciplinary boundaries?

---

## 🛠 Methodology & Tech Stack

The project employs a **hybrid approach** combining Top-Down (Dictionary-based) and Bottom-Up (LDA) methods:

### Data Processing Pipeline
- **PDF Extraction**: [GROBID](https://grobid.readthedocs.io/) converts academic PDFs → structured XML/TEI format
- **Text Preprocessing**: spaCy performs tokenization, lemmatization, POS tagging, and stop-word filtering
- **Dictionary-Based Analysis**: Expert-curated domain lexicons quantify keyword density and disciplinary weight
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) extracts latent thematic structures

### Technologies
- **Python 3.8+**
- **spaCy** — NLP preprocessing
- **scikit-learn** — LDA topic modeling
- **pandas** — Data aggregation & analysis
- **GROBID** — Academic PDF parsing

---

## 📊 Key Outputs

The analysis generates the following CSV reports:

1. **`lda_N_topics.csv`** — Top 20 terms per topic with weights
2. **`lda_N_doc_topic_distribution.csv`** — Per-article topic distribution matrix
3. **`neanderthal_detailed_analysis.csv`** — Domain-level keyword counts and relative frequencies
4. **`neanderthal_count_vectorizer.csv`** — Fine-grained keyword occurrence data

---

## 📁 Repository Structure

```
NLP_PROJECT/
├── text_analysis_course_project.py  # Main pipeline script
├── import spacy.py                   # spaCy model import check
├── test_extraction.py                # Unit tests for PDF extraction
├── config.json                       # Configuration settings
├── corpus_clean_texts.json           # Preprocessed corpus (auto-generated)
├── articles/                         # Source PDFs (39 academic papers)
├── xml_results/                      # GROBID-extracted TEI/XML outputs
├── tmp/                              # Temporary processing files
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- A running GROBID server (default: `http://localhost:8070`)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/neanderthal-language-analysis.git
   cd NLP_PROJECT
   ```

2. **Create and activate virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Start GROBID server** (Docker)
   ```bash
   docker run -t --rm -p 8070:8070 lfoppiano/grobid:latest
   ```

### Usage

**Full pipeline (PDF extraction → preprocessing → analysis)**
```bash
python text_analysis_course_project.py articles
```

**Topic modeling only** (from saved corpus)
```bash
python text_analysis_course_project.py --topic-only 4
```

Where `4` is the number of topics to extract.

---

## 📈 Analysis Workflow

```
Academic PDFs (39 articles, 1989–2025)
        ↓
   GROBID Parser
        ↓
   XML/TEI Output
        ↓
   Text Extraction (remove figures/notes)
        ↓
   NLP Preprocessing (spaCy)
        ↓
   Lemmatization + Stop-word Removal
        ↓
   ├─→ Dictionary-Based Analysis (domain counts)
   │
   └─→ LDA Topic Modeling
        ↓
   CSV Reports (analysis outputs)
```

---

## 📝 Domain Lexicons

Three expert-curated disciplinary dictionaries guide the analysis:

- **Anatomy**: hyoid, larynx, pharynx, vocal fold, basilar membrane, endocast, Broca, Wernicke, etc.
- **Genetics**: FOXP2, CNTnap2, admixture, introgression, genome, SNP, mutation, etc.
- **Archaeology/Behavior**: ornament, burial, tool, pigment, ochre, ritual, hafting, Mousterian, etc.

---

## 📚 Key References

- Green et al. (2010). "A Draft Sequence of the Neandertal Genome"
- Hublin (2009). "The Origin of Neandertals"
- Dediu & Levinson (2013). "On the Antiquity of Language"
- Berwick et al. (2013). "Neanderthal Language: Just-so Stories Take Center Stage"

---

## 👤 Author

**Guy Schrift**  
Department of Archaeology and Ancient Near Eastern Cultures  
Tel Aviv University  

---

## 📄 License

This project is open-source. Please cite appropriately if using in academic work.

---

## 🔧 Troubleshooting

### GROBID Connection Issues
- Ensure Docker container is running: `docker ps`
- Check server health: `http://localhost:8070/api/isalive`
- Increase timeout if PDFs are large: modify `timeout=120` in `run_full_analysis()`

### Out-of-Memory Errors
- Reduce number of PDFs processed per batch
- Increase available RAM or use streaming LDA

---

## 📧 Contact & Support

For questions or contributions, contact Guy Schrift or open an issue on GitHub.
