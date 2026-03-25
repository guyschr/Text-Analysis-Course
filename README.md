Analysis of Disciplinary Weight in Neanderthal Language Research
Project Overview

This project applies Natural Language Processing (NLP) and Computational Linguistics to analyze 39 key academic articles (1989–2025) concerning the linguistic capabilities of Neanderthals. 
The goal is to quantify the shifting "weight" of evidence across three primary domains: Anatomy, Genetics, and Archaeology/Behavioral.
Research Questions

    Disciplinary Weight: How has the prominence of different scientific fields shifted in the debate over Neanderthal language?

    Topic Modeling: Can unsupervised machine learning (LDA) identify thematic clusters that align with traditional disciplinary boundaries?

Methodology & Tech Stack

The project uses a hybrid Top-Down (Dictionary) and Bottom-Up (LDA) approach:

    PDF Parsing:  for converting academic PDFs into structured XML/TEI and raw text.

    Preprocessing:  for tokenization, lemmatization, and stop-word removal.

    Dictionalry based Analysis: Custom Python scripts to measure keyword density based on expert-curated disciplinary dictionaries.

    Topic Modeling:  to extract 3 latent themes across the corpus.

Key Findings


Repository Structure
How to Use

    Clone the repo: git clone https://github.com/your-username/neanderthal-language-analysis.git

    Install dependencies: pip install -r requirements.txt

    Run the analysis: 

Author

Guy Schrift
Department of Archaeology and Ancient Near Eastern Cultures
Tel Aviv University
