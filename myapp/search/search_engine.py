from myapp.search.objects import Document
from myapp.search.algorithms import build_search_index, search_in_corpus


class SearchEngine:
    """Class that implements the search engine logic"""

    def __init__(self):
        self.index_data = None

    def search(self, search_query: str, search_id: str, corpus: dict, top_k: int = 20):
        print("Search query:", search_query)

        if self.index_data is None:
            self.index_data = build_search_index(corpus)

        ranked_results = search_in_corpus(search_query, self.index_data, top_k=top_k)

        results = []
        for pid, score in ranked_results:
            doc = corpus[pid]
            results.append(
                Document(
                    pid=doc.pid,
                    title=doc.title,
                    description=doc.description,
                    selling_price=doc.selling_price,
                    discount=doc.discount,
                    average_rating=doc.average_rating,
                    brand=doc.brand,
                    out_of_stock=doc.out_of_stock,
                    images=doc.images,
                    # Internal link for click tracking -> doc_details, while keeping original URL accessible if needed
                    url=f"doc_details?pid={doc.pid}&search_id={search_id}&param2=2",
                    ranking=round(score, 4),
                )
            )
        return results