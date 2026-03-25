import logging
import os
import sys
from xml.dom import minidom
from datetime import datetime
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from types import SimpleNamespace
from grobid_client.grobid_client import GrobidClient
import xml.etree.ElementTree as ET
import json

# Configure logger to print to console and write to a file in the project folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(os.path.dirname(__file__), f"processing_{timestamp}.log")
logger = logging.getLogger("nlp_project")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)



# 1. SETUP: Models and Stop Words
nlp = spacy.load("en_core_web_sm")

# Add archaeological shortcuts to stop words 
custom_stops = {"ky", "kya", "mya", "ka", "fig", "table", "et", "al", "pp", "sh", "la", "neanderthal", "neandertal", "language"}
for word in custom_stops:
    nlp.Defaults.stop_words.add(word)
    nlp.vocab[word].is_stop = True

# 2. DICTIONARIES (Lowercase for matching)
dictionaries = {
     "Anatomy": [ 
        "hyoid", "larynx", "pharynx", "svt", "basicranium", 
        "vocal fold", "tongue root", "oral cavity", "mandible", 
        "maxilla", "quantal vowels", "articulation", "f1", "f2"
        "ossicle", "malleus", "incus", "stapes", "cochlea","ear" 
        "middle ear", "inner ear", "tympanic", "bandwidth", 
        "hearing sensitivity", "audiogram", "frequency"
        "endocast", "broca", "wernicke", "frontal lobe", 
        "temporal lobe", "cerebral asymmetry", "neuroanatomy"
        ],
    "Behavior/Archaeology":[
         "ornament", "bead", "pendant", "perforated shell", "ochre", 
        "pigment", "hematite", "manganese", "raptor talon", "jewelry", 
        "engraving", "incised", "cave art", "parietal", "figurative",
        "burial", "grave goods", "funerary", "modernity", "ritual", 
        "composite tool", "birch pitch", "hafting", "hafted", 
        "mousterian", "chatelperronian", "levallois", "lithic", 
        "chaine operatoire", "tactical hunting", "social brain",
        "hearth", "site structure", "long-distance", "raw material", 
        "trade", "exchange", "demography", "cultural transmission"
        ],
    "Genetics": [ 
        "foxp2", "cntnap2", "nfix", "admixture", "introgression", 
        "genome", "sequencing", "ancient dna", "adna", "genotype",
        "allele", "snp", "nucleotide", "haplotype", "transcriptome", 
        "epigenetic", "methylation", "selective sweep", "purifying selection", 
        "transcription factor", "base pair", "heterozygous", "homozygous",
        "mutation", "variant", "gene expression", "regulation", 
        "homology", "lineage", "phylogenetic", "divergence", 
        "coding sequence", "non-coding", "regulatory element", 
        "enhancer", "promoter", "locus", "proteomics",
        "hominin", "denisovan", "mrca", "genetic distance", 
        "derived", "ancestral", "hybridization", 
        "population genetics", "genetic drift", "bottleneck",
        ]
}

def parse_filename(text):
    # Split the string by '-' but only up to 2 times
    # This prevents names or titles with hyphens from being split further
    parts = text.split("-", 2)

    # Check if we have all three parts
    if len(parts) == 3:
        year = parts[0].strip()          # Part before the 1st hyphen
        author = parts[1].strip()        # Part between the 1st and 2nd hyphen
        article_name = parts[2].strip()  # Part after the 2nd hyphen

        return {
            "year": year,
            "author": author,
            "article_name": article_name
        }
    else:
        return "The text does not contain at least 2 hyphens."

def handle_topic_modeling(corpus_cleaned_text, no_of_topics=None):
    if isinstance(corpus_cleaned_text, str):
        # Load from file
        with open(corpus_cleaned_text, 'r') as f:
            corpus_cleaned_text = json.load(f)

    # If number of topics is not provided or is empty, default to 3
    if not no_of_topics:
        no_of_topics = 3

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    # Extract filenames and text strings from corpus objects
    corpus_filenames = []
    corpus_texts = []
    
    for item in corpus_cleaned_text:
        if isinstance(item, dict):
            corpus_filenames.append(item["filename"])
            corpus_texts.append(item["clean_text"])
        else:
            # Fallback for legacy string format
            corpus_filenames.append(f"article_{len(corpus_filenames)}")
            corpus_texts.append(item)
    
    # 1. Vectorize the entire corpus (all 39 articles)
    # limit max_df to 0.8 to ignore words appearing in >80% of papers (too common)
    # set min_df to 2 to ignore words appearing in only one paper (noise)
    vectorizer = CountVectorizer(max_df=0.8, min_df=2);
    dtm = vectorizer.fit_transform(corpus_texts)  

    # 2. Initialize LDA
    # n_components is controlled by no_of_topics
    lda = LatentDirichletAllocation(n_components=no_of_topics, random_state=42)
    lda.fit(dtm)

    # 3. Display the discovered "Weights" (Top words per topic)
    lda_data = []
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        logger.info(f"Topic #{topic_idx + 1}:")
        logger.info([words[i] for i in topic.argsort()[:-11:-1]]) # Top 10 words
        lda_data.append({
            "topic": topic_idx + 1,
            "top_words": [words[i] for i in topic.argsort()[:-11:-1]],
            "weights": topic[topic.argsort()[:-11:-1]].tolist()
        })

    df_lda = pd.DataFrame(lda_data)
    out_file = f"lda_{no_of_topics}_topics.csv"
    df_lda.to_csv(out_file, index=False)
    logger.info("LDA topic modeling completed and saved to %s", out_file)

    # 4. Extract per-document topic distribution
    doc_topic_dist = lda.transform(dtm)
    
    # 5. Create dataframe with article names and topic distributions
    topic_cols = [f"topic_{i+1}" for i in range(no_of_topics)]
    df_doc_dist = pd.DataFrame(doc_topic_dist, columns=topic_cols)
    df_doc_dist.insert(0, "filename", corpus_filenames)
    
    # Parse filename to extract year and author
    df_doc_dist["year"] = df_doc_dist["filename"].apply(
        lambda f: parse_filename(f)["year"] if isinstance(parse_filename(f), dict) else "Unknown"
    )
    df_doc_dist["author"] = df_doc_dist["filename"].apply(
        lambda f: parse_filename(f)["author"] if isinstance(parse_filename(f), dict) else "Unknown"
    )
    df_doc_dist["article_name"] = df_doc_dist["filename"].apply(
        lambda f: parse_filename(f)["article_name"] if isinstance(parse_filename(f), dict) else "Unknown"
    )
    
    # Calculate dominant topic and its weight
    df_doc_dist["dominant_topic"] = df_doc_dist[topic_cols].idxmax(axis=1).str.replace("topic_", "").astype(int)
    df_doc_dist["dominant_topic_weight"] = df_doc_dist[topic_cols].max(axis=1)
    
    # Reorder columns for clarity
    cols_order = ["filename", "year", "author", "article_name", "dominant_topic", "dominant_topic_weight"] + topic_cols
    df_doc_dist = df_doc_dist[cols_order]
    
    doc_dist_file = f"lda_{no_of_topics}_doc_topic_distribution.csv"
    df_doc_dist.to_csv(doc_dist_file, index=False)
    logger.info("Document-topic distribution saved to %s", doc_dist_file)
    
    return df_lda, df_doc_dist



def handle_count_vectorizer(clean_text_string, filename, dictionaries):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    
    count_vectorizer_results = []
    
    # Pre-process the text once to save time
    text_list = [clean_text_string]
    parse_filename_result = parse_filename(filename)

    # 1. Initialize Vectorizer without a fixed vocabulary
    # Since text_list is one article and already cleaned:
    # We use a simple count without further preprocessing.
    vectorizer = CountVectorizer(analyzer='word', token_pattern=r'(?u)\b\w\w+\b')
    try:
        # 1. Fit and transform the single article
        # X will be a sparse matrix of shape (1, number_of_unique_words)        
        X = vectorizer.fit_transform(text_list)
        feature_names = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        # Calculate total words in this cleaned article for normalization
        total_cleaned_words = sum(counts)

        # 2. Iterate through all discovered words
        for idx, kw in enumerate(feature_names):
            count_val = int(counts[idx])

            # 3. Categorize the word based dictionaries
            assigned_domain = "Other/General"
            for domain, keywords in dictionaries.items():
                if kw in keywords:
                    assigned_domain = domain
                    break

            # 4. Append results (including 'Other' for a complete weight analysis)
            count_vectorizer_results.append({
                "article_name": parse_filename_result["article_name"],
                "published_year": parse_filename_result["year"],
                "author": parse_filename_result["author"],
                "domain": assigned_domain,
                "keyword": kw,
                "count": count_val,
                "relative_frequency": count_val / total_cleaned_words if total_cleaned_words > 0 else 0
            })
            # Log finding for transparency
            # logger.info(f"CountVectorizer found: {filename} | {assigned_domain} | {kw}: {count_val} (Relative Frequency: {count_val / total_cleaned_words:.6f})")
    except Exception as e:
        logger.error(f"Error processing {parse_filename_result['article_name']}: {e}")
          

    return count_vectorizer_results


def handle_dictionary_count_vectorizer(clean_text_string, filename, dictionaries):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    
    count_vectorizer_results = []
    
    # Pre-process the text once to save time
    text_list = [clean_text_string]

    parse_filename_result = parse_filename(filename)

    for domain, keywords in dictionaries.items():
        # 1. Initialize with the domain-specific vocabulary
        vectorizer = CountVectorizer(vocabulary=keywords)
        
        # 2. Transform the text
        try:
            X = vectorizer.fit_transform(text_list)
            feature_names = vectorizer.get_feature_names_out()
            counts = X.toarray()[0]
            
            # 3. Map counts to keywords
            for idx, kw in enumerate(feature_names):
                count_val = int(counts[idx])
                if count_val > 0:  # Only store keywords that actually appeared
                    count_vectorizer_results.append({
                        "article_name": parse_filename_result["article_name"],
                        "published_year": parse_filename_result["year"],
                        "author": parse_filename_result["author"],
                        "domain": domain,
                        "keyword": kw,
                        "count": count_val
                    })
                    # Log finding for transparency
                    # logger.info(f"Match: {filename} | {domain} | {kw}: {count_val}")
        
        except ValueError:
            # Handle cases where no keywords from the dictionary are found
            continue

    return count_vectorizer_results

def handle_dictionary_analysis(clean_text_string, total_words, filename):
    results = []
    parse_filename_result = parse_filename(filename)
    for domain, keywords in dictionaries.items():
        count = 0
        for kw in keywords:
            kw_count = clean_text_string.count(kw)
            if kw_count > 0:
                count += kw_count
                # logger.info("Counting keyword '%s' in domain '%s': current count %d", kw, domain, kw_count)
        weight = count / total_words if total_words > 0 else 0
        results.append({
            "article_name": parse_filename_result["article_name"],
            "published_year": parse_filename_result["year"],
            "author": parse_filename_result["author"],
            "domain": domain,
            "count": count,
            "total_words": total_words,
            "weight": weight
        })

        # for a in results:
        #     logger.info("Article: %s | Year: %s | Domain: %s | Count: %d| Total Words: %d| Weight: %.6f", 
        #                 a['article_name'], a['published_year'], a['domain'], a['count'], a['total_words'], a['weight'])

    return results

def process_single_article(pdf_path, client, filename):
    # Step A: GROBID Extraction
    _, status, xml_out = client.process_pdf(
        "processFulltextDocument", pdf_path, 
        generateIDs=False, consolidate_header=True, 
        segment_sentences=True, consolidate_citations=True,
        include_raw_citations=False, include_raw_affiliations=False,
        tei_coordinates=False
    )
    
    if status != 200: return None

    # Step B: Pruning Figures/Notes & Extracting Text
    root = ET.fromstring(xml_out)
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    body = root.find(".//tei:body", ns)
    if body is None: return None
    
    # Robust removal of <figure> and <note> (namespace-aware)
    for tag in ['figure', 'note']:
        full_tag = f"{{{ns['tei']}}}{tag}"
        for parent in body.iter():
            for child in list(parent):            # list(...) to avoid mutation during iteration
                if child.tag == full_tag:
                    parent.remove(child)
        
    raw_text = " ".join(["".join(p.itertext()) for p in body.findall(".//tei:p", ns)])
    raw_word_count = len(raw_text.split())
    # Step C: NLP Preprocessing (Lemmatization & Stop Words)
    doc = nlp(raw_text.lower())
    # We create a 'cleaned string' of lemmas to preserve sequence for phrase matching
    clean_lemmas = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.is_alpha and len(t.lemma_) > 1 and t.lemma_ not in custom_stops]
    
    total_words = len(clean_lemmas)
    clean_text_string = " ".join(clean_lemmas)
    # logger.info("extracted lemmas: %s", clean_lemmas)
    logger.info("Total clean words: %d", total_words)
    
    # Step D: Lexical Analysis (Phrase-Aware)
    dictionary_analysis_results = handle_dictionary_analysis(clean_text_string, total_words, filename)
    
    # Step E: CountVectorizer Analysis 
    # count_vectorizer_results = handle_dictionary_count_vectorizer(clean_text_string, filename, dictionaries)
    # count_vectorizer_results = handle_count_vectorizer(clean_text_string, filename, dictionaries)
    count_vectorizer_results = [] 
    return dictionary_analysis_results, count_vectorizer_results, clean_text_string,  raw_word_count

# 3. MAIN EXECUTION LOOP
def run_full_analysis(folder_path):
    client = GrobidClient(config_path=None, timeout=120)
    all_rows_dictionary = [] # This will hold the dictionary data for ALL articles
    all_rows_count_vectorizer = [] # This will hold the count vectorizer data for ALL articles
    corpus_clean_text_strings = [] # This will hold the cleaned text strings for ALL articles
    raw_word_counts = [] # This will hold the raw word counts for ALL articles

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            logger.info("Processing article: %s", filename)
            dictionary_analysis_results, count_vectorizer_results, clean_text_string,  raw_word_count = process_single_article(os.path.join(folder_path, filename), client, filename)

            if dictionary_analysis_results is not None:
               all_rows_dictionary.extend(dictionary_analysis_results) # 'extend' flattens the list of namespaces
            if count_vectorizer_results is not None:
                all_rows_count_vectorizer.extend(count_vectorizer_results) # 'extend' flattens the list of namespaces
            # Store corpus as objects with filename and clean text
            corpus_clean_text_strings.append({
                "filename": filename,
                "clean_text": clean_text_string
            })
            raw_word_counts.append(raw_word_count)

    logger.info("All articles processed. Total articles: %d", len(corpus_clean_text_strings))
    logger.info("Raw word counts: %s", raw_word_counts)

    # Save the corpus for reuse
    with open('corpus_clean_texts.json', 'w') as f:
        json.dump(corpus_clean_text_strings, f)
    logger.info("Corpus saved to corpus_clean_texts.json")

    # Run topic modeling with the corpus
    handle_topic_modeling(corpus_clean_text_strings, no_of_topics=3)

    # Create the DataFrame
    df_dictionary = pd.DataFrame(all_rows_dictionary)  # This will have the dictionary analysis results
    df_count_vectorizer = pd.DataFrame(all_rows_count_vectorizer)  # This will have the count vectorizer results


    # Reorder columns to match your exact requirement and save to CSV
    if not df_dictionary.empty:
        df_dictionary = df_dictionary[['article_name', 'published_year','author', 'domain', 'count', 'total_words',
                                       'weight']]
        df_dictionary.to_csv("neanderthal_detailed_analysis.csv", index=False)
 
    if not df_count_vectorizer.empty: 
        df_count_vectorizer = df_count_vectorizer[['article_name', 'published_year','author', 'domain', 'keyword',
                                                    'count','relative_frequency']]
        df_count_vectorizer.to_csv("neanderthal_count_vectorizer.csv", index=False)

    logger.info("dictionary analysis CSV generated successfully with %d rows.", len(df_dictionary))
    logger.info("count vectorizer CSV generated successfully with %d rows.", len(df_count_vectorizer))
    return df_dictionary, df_count_vectorizer

# To run it:
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--topic-only":
        # Expect the next argument to be the number of topics
        if len(sys.argv) < 3:
            logger.error("Usage: python text_analysis_course_project.py --topic-only <num_topics>")
            sys.exit(1)
        try:
            n_topics = int(sys.argv[2])
        except ValueError:
            logger.error("<num_topics> must be an integer (e.g., 3).")
            sys.exit(1)

        logger.info("Running topic modeling only from saved corpus (topics=%d)", n_topics)
        handle_topic_modeling('corpus_clean_texts.json', no_of_topics=n_topics)
    else:
        final_df = run_full_analysis(f"./articles")