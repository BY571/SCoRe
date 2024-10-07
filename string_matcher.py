import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)


class LLMAnswerComparator:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def tfidf_similarity(self, text1, text2):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def bert_similarity(self, text1, text2):
        embeddings = self.bert_model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def compare(self, response, oracle_response, method="ensemble"):
        """
        Compare an LLM response with an oracle (reference) response using the specified method.

        Args:
        response (str): The LLM-generated response to evaluate.
        oracle_response (str): The reference (correct) response.
        method (str): The comparison method to use ('tfidf', 'bert', or 'ensemble').

        Returns:
        tuple: (similarity score, boolean indicating if the response is considered correct)
        """
        if method == "tfidf":
            similarity = self.tfidf_similarity(response, oracle_response)
        elif method == "bert":
            similarity = self.bert_similarity(response, oracle_response)
        elif method == "ensemble":
            tfidf_sim = self.tfidf_similarity(response, oracle_response)
            bert_sim = self.bert_similarity(response, oracle_response)
            similarity = np.mean([tfidf_sim, bert_sim])
        else:
            raise ValueError("Invalid method. Choose 'tfidf', 'bert', or 'ensemble'.")

        is_correct = similarity >= self.threshold
        return similarity, is_correct

    def batch_compare(self, responses, oracle_responses, method="ensemble"):
        """
        Perform batch comparison of multiple LLM responses with their corresponding oracle responses.

        Args:
        responses (list): List of LLM-generated responses to evaluate.
        oracle_responses (list): List of corresponding reference (correct) responses.
        method (str): The comparison method to use ('tfidf', 'bert', or 'ensemble').

        Returns:
        list: List of tuples, each containing (similarity score, boolean indicating if the response is considered correct)
        """
        if len(responses) != len(oracle_responses):
            raise ValueError(
                "The number of responses and oracle responses must be the same."
            )

        results = []
        for response, oracle_response in zip(responses, oracle_responses):
            results.append(self.compare(response, oracle_response, method))

        return results


# Example usage
if __name__ == "__main__":
    comparator = LLMAnswerComparator(threshold=0.8)

    # Single comparison
    llm_response = "The capital of France is Paris."
    oracle_response = "Paris is the capital of France."
    score, correct = comparator.compare(llm_response, oracle_response, method="bert")
    print(f"Single comparison - Similarity score: {score:.2f}, Is correct: {correct}")

    # Batch comparison
    llm_responses = [
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "The chemical symbol for gold is Au.",
    ]
    oracle_responses = [
        "Paris is the capital of France.",
        "Jupiter is the largest planet in our solar system.",
        "Au is the chemical symbol for gold.",
    ]
    batch_results = comparator.batch_compare(
        llm_responses, oracle_responses, method="ensemble"
    )
    for i, (score, correct) in enumerate(batch_results):
        print(
            f"Batch comparison {i+1} - Similarity score: {score:.2f}, Is correct: {correct}"
        )
