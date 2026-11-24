import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
You are an expert e-commerce product advisor helping users choose the best option from retrieved products.

## Strict Grounding Rules:
1. Base your answer ONLY on the product attributes provided below (Price, Discount, Rating, Brand, Stock Status, Description, and Specs). NEVER invent prices, materials, or features that are not listed.
2. Prefer recommending products that are "In Stock". If the top match is "Out of Stock", explicitly warn the user and recommend an in-stock alternative.
3. If no product is a good fit for the user's request, return ONLY this exact phrase:
   "There are no good products that fit the request based on the retrieved results."

## Output Format:
- Best Product: [Product PID] [Product Name] (Price, Rating)
- Why: [Explain in plain language why this product is the best fit using its actual price, discount, rating, brand, or description.]
- Best Value / Alternative (optional): [Briefly mention a strong budget/discounted or alternative in-stock product if available.]

## Retrieved Products:
{retrieved_results}

## User Request:
{user_query}
"""

    def _format_product(self, res) -> str:
        """
        IMPROVEMENT 1: Format rich product metadata (price, discount, rating, brand,
        stock status, and description) instead of only PID and Title.
        """
        price = f"₹{res.selling_price:.2f}" if res.selling_price is not None else "N/A"
        discount = f"{int(res.discount)}% OFF" if res.discount else "No discount"
        rating = f"{res.average_rating}/5" if res.average_rating is not None else "No ratings"
        brand = res.brand if res.brand else "N/A"
        stock = "Out of Stock" if res.out_of_stock else "In Stock"
        desc = (res.description[:200] + "...") if res.description and len(res.description) > 200 else (res.description or "N/A")

        return (
            f"- PID: {res.pid} | Title: {res.title} | Brand: {brand} | Status: {stock} | "
            f"Price: {price} ({discount}) | Rating: {rating} | Description: {desc}"
        )

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 5):
        """
        Generate a grounded product recommendation using the top retrieved search results.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."

        # IMPROVEMENT 2: Handle "no good products" / empty retrieval immediately without wasting an API call
        if not retrieved_results:
            return "There are no good products that fit the request based on the retrieved results."

        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

            # Format top-N products with full metadata
            formatted_results = "\n".join(
                [self._format_product(res) for res in retrieved_results[:top_N]]
            )

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query
            )

            # IMPROVEMENT 3: Lower temperature (0.2) + stricter grounding prompt on top-5 ranked results
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                temperature=0.2,
                max_tokens=300,
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER