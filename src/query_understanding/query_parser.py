import re
from src.domain.query import Query

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "about", "using", 
    "paper", "papers", "article", "articles", "study", "studies", "research", 
    "on", "of", "to", "in", "a", "an", "is", "by", "as", "at", "or", "how", 
    "what", "which", "where"
}

def parse_query(query: Query) -> Query:
    raw_text = query.text
    lower_text = raw_text.lower().strip()
    
    metadata_constraints = {}
    
    # 1. Extract publication year constraints
    # Pattern: "between 2020 and 2025"
    between_match = re.search(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", lower_text)
    if between_match:
        y1, y2 = int(between_match.group(1)), int(between_match.group(2))
        metadata_constraints["year"] = {">=": y1, "<=": y2}
        lower_text = lower_text.replace(between_match.group(0), "")
        
    # Pattern: "after 2023", "since 2023", "post 2023"
    after_match = re.search(r"\b(after|since|post)\s+(\d{4})\b", lower_text)
    if after_match:
        year = int(after_match.group(2))
        metadata_constraints["year"] = {">": year}
        lower_text = lower_text.replace(after_match.group(0), "")
        
    # Pattern: "before 2025", "prior to 2025", "pre 2025"
    before_match = re.search(r"\b(before|prior\s+to|pre)\s+(\d{4})\b", lower_text)
    if before_match:
        year = int(before_match.group(2))
        metadata_constraints["year"] = {"<": year}
        lower_text = lower_text.replace(before_match.group(0), "")
        
    # Pattern: "recent"
    if "recent" in lower_text:
        metadata_constraints["year"] = {">=": 2024}  # 2 years before current year 2026
        lower_text = re.sub(r"\brecent\b", "", lower_text)

    # 2. Extract intent (document style / type)
    intent = "general"
    
    # survey, review, overview, tutorial, state of the art
    survey_patterns = [r"\bsurvey\b", r"\breview\b", r"\boverview\b", r"\btutorial\b", r"\bstate\s+of\s+the\s+art\b"]
    if any(re.search(p, lower_text) for p in survey_patterns):
        intent = "survey"
        for p in survey_patterns:
            lower_text = re.sub(p, "", lower_text)
            
    # vs, versus, comparison, compare
    comparison_patterns = [r"\bvs\b", r"\bversus\b", r"\bcomparison\b", r"\bcompare\b"]
    if any(re.search(p, lower_text) for p in comparison_patterns):
        intent = "comparison"
        for p in comparison_patterns:
            lower_text = re.sub(p, "", lower_text)
            
    # dataset, benchmark, corpus, datasets
    dataset_patterns = [r"\bdataset\b", r"\bbenchmark\b", r"\bcorpus\b", r"\bdatasets\b"]
    if any(re.search(p, lower_text) for p in dataset_patterns):
        intent = "dataset"
        for p in dataset_patterns:
            lower_text = re.sub(p, "", lower_text)

    # 3. Boost & Exclude terms
    boost_terms = []
    # Extract quoted phrases as boost terms
    quoted_phrases = re.findall(r'"([^"]+)"', raw_text)
    for q in quoted_phrases:
        boost_terms.append(q.lower().strip())
        lower_text = lower_text.replace(f'"{q.lower()}"', "")
        
    exclude_terms = []
    # Match words preceded by minus "-" e.g. "-neural"
    minus_matches = re.findall(r'-\b\w+\b', raw_text)
    for m in minus_matches:
        term = m[1:].lower().strip()
        exclude_terms.append(term)
        lower_text = lower_text.replace(m.lower(), "")
        
    # Match words preceded by "not" e.g. "not neural"
    not_matches = re.findall(r'\bnot\s+(\w+)\b', lower_text)
    for term in not_matches:
        exclude_terms.append(term)
        lower_text = lower_text.replace(f"not {term}", "")

    # 4. Clean up topic text and extract keywords
    fluff = ["papers", "paper", "articles", "article", "studies", "study", "research"]
    for f in fluff:
        lower_text = re.sub(r'\b' + f + r'\b', "", lower_text)
        
    cleaned_text = " ".join(lower_text.split())
    cleaned_text = re.sub(r'^[,\s\-:]+|[,\s\-:]+$', '', cleaned_text).strip()
    
    words = re.findall(r"\b\w+\b", cleaned_text)
    keywords = [w for w in words if w not in STOPWORDS and not w.isdigit()]

    return Query(
        text=query.text,
        top_k=query.top_k,
        filters=query.filters,
        paper_id=query.paper_id,
        section=query.section,
        metadata=query.metadata,
        topic=cleaned_text,
        keywords=keywords,
        intent=intent,
        boost_terms=boost_terms,
        exclude_terms=exclude_terms,
        metadata_constraints=metadata_constraints
    )
