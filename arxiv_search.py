# Pranay Teja Chintakunta - 25079476
# MSc AI & ML - University of Limerick
# Module 3 - Live Paper Search (ArXiv, PubMed, CrossRef) + Alerts

import arxiv
import os
import urllib.request

class ArxivSearcher:

    def __init__(self):
        self.client = arxiv.Client(
            page_size=20,
            delay_seconds=1.0,
            num_retries=1
        )
        print("ArXiv searcher ready")

    def search_papers(self, query, max_results=5, from_year=None, to_year=None):
        import concurrent.futures

        def _do_search():
            fetch_count = max_results * 3 if (from_year or to_year) else max_results
            search = arxiv.Search(
                query=query,
                max_results=fetch_count,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = []
            for paper in self.client.results(search):
                year = paper.published.year
                if from_year and year < int(from_year):
                    continue
                if to_year and year > int(to_year):
                    continue
                results.append({
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "pdf_url": paper.pdf_url,
                    "arxiv_id": paper.get_short_id(),
                    "categories": paper.categories
                })
                if len(results) >= max_results:
                    break
            return results

        try:
            # give ArXiv max 20 seconds, then give up (prevents freezing the backend)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_do_search)
                return future.result(timeout=20)
        except concurrent.futures.TimeoutError:
            print("arxiv search timed out")
            return []
        except Exception as e:
            print(f"search error: {e}")
            return []

    def download_paper(self, pdf_url, arxiv_id, save_folder="uploaded_papers"):
        try:
            os.makedirs(save_folder, exist_ok=True)
            filename = arxiv_id.replace("/", "_") + ".pdf"
            save_path = os.path.join(save_folder, filename)
            urllib.request.urlretrieve(pdf_url, save_path)
            print(f"downloaded {filename}")
            return save_path
        except Exception as e:
            print(f"download error: {e}")
            return None

class PubMedSearcher:

    def __init__(self):
        from Bio import Entrez
        Entrez.email = "25079476@studentmail.ul.ie"
        self.Entrez = Entrez
        print("PubMed searcher ready")

    def search_papers(self, query, max_results=5, from_year=None, to_year=None):
        try:
            # add a year-range filter to the query if provided ([dp] = date of publication)
            term = query
            if from_year and to_year:
                term = f"{query} AND {from_year}:{to_year}[dp]"
            elif from_year:
                term = f"{query} AND {from_year}:3000[dp]"
            elif to_year:
                term = f"{query} AND 1800:{to_year}[dp]"

            handle = self.Entrez.esearch(
                db="pubmed",
                term=term,
                retmax=max_results,
                sort="relevance"
            )
            record = self.Entrez.read(handle)
            handle.close()
            id_list = record.get("IdList", [])

            if not id_list:
                return []

            handle = self.Entrez.efetch(
                db="pubmed",
                id=",".join(id_list),
                rettype="abstract",
                retmode="xml"
            )
            papers_data = self.Entrez.read(handle)
            handle.close()

            results = []
            for article in papers_data.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    art = medline["Article"]
                    title = str(art.get("ArticleTitle", "No title"))

                    abstract = "No abstract available"
                    if "Abstract" in art:
                        abstract_parts = art["Abstract"].get("AbstractText", [])
                        abstract = " ".join([str(p) for p in abstract_parts])

                    authors = []
                    for author in art.get("AuthorList", []):
                        last = author.get("LastName", "")
                        fore = author.get("ForeName", "")
                        if last:
                            authors.append(f"{fore} {last}".strip())

                    year = "Unknown"
                    pub_date = art.get("Journal", {}).get(
                        "JournalIssue", {}).get("PubDate", {})
                    if "Year" in pub_date:
                        year = str(pub_date["Year"])

                    pmid = str(medline.get("PMID", "unknown"))

                    results.append({
                        "title": title,
                        "authors": authors,
                        "summary": abstract,
                        "published": year,
                        "pmid": pmid,
                        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
                except Exception as e:
                    print(f"error parsing article: {e}")
                    continue

            return results
        except Exception as e:
            print(f"pubmed search error: {e}")
            return []

    def save_abstract_as_text(self, paper, save_folder="uploaded_papers"):
        import os
        try:
            os.makedirs(save_folder, exist_ok=True)
            filename = "pubmed_" + paper["pmid"] + ".txt"
            save_path = os.path.join(save_folder, filename)
            content = f"""Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Year: {paper['published']}
PubMed URL: {paper['pubmed_url']}

Abstract:
{paper['summary']}
"""
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"saved {filename}")
            return save_path
        except Exception as e:
            print(f"save error: {e}")
            return None

class CrossRefSearcher:

    def __init__(self):
        self.base_url = "https://api.crossref.org/works"
        print("CrossRef searcher ready")

    def search_papers(self, query, max_results=5, from_year=None, to_year=None):
        import requests
        try:
            params = {
                "query": query,
                "rows": max_results,
                "sort": "relevance",
                "order": "desc"
            }
            # add year-range filter if provided
            filters = []
            if from_year:
                filters.append(f"from-pub-date:{from_year}-01-01")
            if to_year:
                filters.append(f"until-pub-date:{to_year}-12-31")
            if filters:
                params["filter"] = ",".join(filters)

            headers = {
                "User-Agent": "RAGResearchAssistant/1.0 (mailto:25079476@studentmail.ul.ie)"
            }
            response = requests.get(
                self.base_url, params=params, headers=headers, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("message", {}).get("items", []):
                title_list = item.get("title", ["No title"])
                title = title_list[0] if title_list else "No title"

                authors = []
                for author in item.get("author", []):
                    given = author.get("given", "")
                    family = author.get("family", "")
                    name = f"{given} {family}".strip()
                    if name:
                        authors.append(name)

                year = "Unknown"
                for date_field in ["published-print", "published-online", "published", "created"]:
                    date_parts = item.get(date_field, {}).get("date-parts", [[None]])
                    if date_parts and date_parts[0] and date_parts[0][0]:
                        candidate = date_parts[0][0]
                        if isinstance(candidate, int) and 1900 <= candidate <= 2026:
                            year = str(candidate)
                            break

                container = item.get("container-title", ["Unknown journal"])
                journal = container[0] if container else "Unknown journal"

                abstract = item.get("abstract", "No abstract available")
                import re
                abstract = re.sub(r"<[^>]+>", "", abstract)

                doi = item.get("DOI", "unknown")

                results.append({
                    "title": title,
                    "authors": authors,
                    "summary": abstract,
                    "published": year,
                    "journal": journal,
                    "doi": doi,
                    "doi_url": f"https://doi.org/{doi}"
                })
            return results
        except Exception as e:
            print(f"crossref search error: {e}")
            return []

    def save_metadata_as_text(self, paper, save_folder="uploaded_papers"):
        import os
        try:
            os.makedirs(save_folder, exist_ok=True)
            safe_doi = paper["doi"].replace("/", "_").replace(".", "_")
            filename = "crossref_" + safe_doi[:20] + ".txt"
            save_path = os.path.join(save_folder, filename)
            content = f"""Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Year: {paper['published']}
Journal: {paper['journal']}
DOI: {paper['doi_url']}

Abstract:
{paper['summary']}
"""
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"saved {filename}")
            return save_path
        except Exception as e:
            print(f"save error: {e}")
            return None

class ResearchAlertSystem:

    def __init__(self):
        self.saved_topics = []
        self.seen_papers = {}
        print("Research alert system ready")

    def add_topic(self, topic):
        if topic and topic not in self.saved_topics:
            self.saved_topics.append(topic)
            self.seen_papers[topic] = set()
            return True
        return False

    def remove_topic(self, topic):
        if topic in self.saved_topics:
            self.saved_topics.remove(topic)
            self.seen_papers.pop(topic, None)
            return True
        return False

    def get_topics(self):
        return self.saved_topics

    def check_for_new_papers(self, topic, max_results=5):
        # alerts stay sorted by newest — that's correct for alerts
        import arxiv
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=topic,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            new_papers = []
            all_papers = []
            for paper in client.results(search):
                paper_id = paper.get_short_id()
                paper_info = {
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors][:3],
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "arxiv_id": paper_id,
                    "pdf_url": paper.pdf_url,
                    "summary": paper.summary[:300]
                }
                all_papers.append(paper_info)
                if topic in self.seen_papers:
                    if paper_id not in self.seen_papers[topic]:
                        new_papers.append(paper_info)
                        self.seen_papers[topic].add(paper_id)
                else:
                    self.seen_papers[topic] = {paper_id}
                    new_papers.append(paper_info)

            return {"new": new_papers, "all": all_papers}
        except Exception as e:
            print(f"alert check error: {e}")
            return {"new": [], "all": []}

if __name__ == "__main__":
    print("=== Testing CrossRef ===")
    cr = CrossRefSearcher()
    papers = cr.search_papers("machine learning healthcare", 3)
    for p in papers:
        print(f"\n{p['title']} ({p['published']})")