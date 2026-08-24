from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult

SURVEY_KEYWORDS = {"survey", "review", "overview", "state of the art", "tutorial", "advances", "recent progress"}

def filter_results(results: list[RetrievalResult], query: Query) -> list[RetrievalResult]:
    filtered = []
    
    year_constraint = query.metadata_constraints.get("year")
    intent = query.intent
    
    for r in results:
        # 1. Check publication year constraints
        if year_constraint:
            doc_year_str = r.metadata.get("year", "N/A")
            try:
                doc_year = int(doc_year_str)
            except (ValueError, TypeError):
                continue
                
            matched = True
            for op, val in year_constraint.items():
                if op == ">" and not (doc_year > val):
                    matched = False
                elif op == "<" and not (doc_year < val):
                    matched = False
                elif op == ">=" and not (doc_year >= val):
                    matched = False
                elif op == "<=" and not (doc_year <= val):
                    matched = False
            if not matched:
                continue
                
        # 2. Check intent-based filtering (e.g. drop non-survey papers if intent is survey)
        if intent == "survey":
            title = r.metadata.get("title", "").lower()
            abstract = r.metadata.get("abstract", "").lower()
            
            kws = r.metadata.get("keywords", [])
            if isinstance(kws, str):
                kws = [k.strip().lower() for k in kws.split(",") if k.strip()]
            elif isinstance(kws, list):
                kws = [str(k).lower() for k in kws]
            else:
                kws = []
                
            has_survey_kw = (
                any(kw in title for kw in SURVEY_KEYWORDS) or
                any(kw in abstract for kw in SURVEY_KEYWORDS) or
                any(kw in kws for kw in SURVEY_KEYWORDS)
            )
            if not has_survey_kw:
                continue
                
        # 3. Check paper_id filter
        if query.paper_id and r.paper_id != query.paper_id:
            continue
            
        # 4. Check section filter
        if query.section and r.section != query.section:
            continue
            
        # 5. Check exclude terms
        if query.exclude_terms:
            text_lower = r.text.lower()
            title_lower = r.metadata.get("title", "").lower()
            abstract_lower = r.metadata.get("abstract", "").lower()
            
            should_exclude = False
            for term in query.exclude_terms:
                if term in text_lower or term in title_lower or term in abstract_lower:
                    should_exclude = True
                    break
            if should_exclude:
                continue

        filtered.append(r)
        
    return filtered
