"""
Multi-Agent System using local Ollama models.

Three agent types:
  1. DeterministicAgent  — pure Python logic, no LLM, always same output
  2. ProbabilisticAgent  — calls LLM, output varies each run (temperature > 0)
  3. RAGAgent            — retrieves from dummy data, then calls LLM with context
  4. OrchestratorAgent   — routes query to right agent(s), merges results

Run:  python agents.py
"""

import json
import random
from openai import OpenAI
from colorama import Fore, Style, init

init(autoreset=True)

# ── Ollama client (works exactly like OpenAI SDK) ──────────────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",          # required by SDK but ignored by Ollama
)

CHAT_MODEL       = "phi4-mini"          # general chat / orchestrator
CODER_MODEL      = "qwen2.5-coder:3b"  # code / technical generation


# ══════════════════════════════════════════════════════════════════
# DUMMY DATA  (replaces a real database or vector store)
# ══════════════════════════════════════════════════════════════════
PRODUCT_CATALOG = {
    "laptop_a": {"name": "ProBook 15",  "price": 55000, "ram": 16, "gpu": False},
    "laptop_b": {"name": "GameRig X",   "price": 85000, "ram": 32, "gpu": True},
    "laptop_c": {"name": "UltraLight S","price": 70000, "ram": 8,  "gpu": False},
}

FAQ_DOCS = [
    {"id": 1, "topic": "return policy",   "text": "You can return any product within 30 days for a full refund."},
    {"id": 2, "topic": "warranty",        "text": "All laptops come with a 1-year manufacturer warranty."},
    {"id": 3, "topic": "shipping",        "text": "Free shipping on orders above Rs 50,000. Delivered in 3-5 days."},
    {"id": 4, "topic": "emi",             "text": "EMI available on 3, 6, 12 month plans with 0% interest on select banks."},
    {"id": 5, "topic": "student discount","text": "Students get 10% off with a valid college ID."},
]

SALES_DATA = [
    {"month": "Jan", "units": 120, "revenue": 6600000},
    {"month": "Feb", "units": 95,  "revenue": 5225000},
    {"month": "Mar", "units": 140, "revenue": 7700000},
]


# ══════════════════════════════════════════════════════════════════
# HELPER — call local LLM
# ══════════════════════════════════════════════════════════════════
def call_llm(prompt: str, model: str = CHAT_MODEL, temperature: float = 0.7,
             system: str = "You are a helpful assistant.") -> str:
    """Call local Ollama model and return the text response."""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        import traceback
        print(f"{Fore.RED}[LLM full error]:\n{traceback.format_exc()}{Style.RESET_ALL}")
        return f"[LLM error: {e}]"


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — DETERMINISTIC
# Pure Python, zero LLM calls. Output is 100% predictable.
# Use for: pricing, inventory checks, rule-based decisions.
# ══════════════════════════════════════════════════════════════════
class DeterministicAgent:
    name = "DeterministicAgent"

    def run(self, query: str) -> dict:
        q = query.lower()

        # Product price lookup
        if "price" in q or "cost" in q or "how much" in q:
            results = []
            for pid, p in PRODUCT_CATALOG.items():
                results.append(f"{p['name']}: Rs {p['price']:,}")
            return {
                "agent": self.name,
                "type": "deterministic",
                "answer": "Current prices:\n" + "\n".join(results),
                "confidence": 1.0,   # always exact
            }

        # Stock / availability
        if "available" in q or "stock" in q or "inventory" in q:
            return {
                "agent": self.name,
                "type": "deterministic",
                "answer": f"{len(PRODUCT_CATALOG)} products in stock.",
                "confidence": 1.0,
            }

        # Sales stats
        if "sales" in q or "revenue" in q or "units" in q:
            total_units   = sum(s["units"]   for s in SALES_DATA)
            total_revenue = sum(s["revenue"] for s in SALES_DATA)
            return {
                "agent": self.name,
                "type": "deterministic",
                "answer": (f"Q1 sales: {total_units} units, "
                           f"Rs {total_revenue:,} revenue."),
                "confidence": 1.0,
            }

        return None   # not my responsibility


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — PROBABILISTIC
# Calls LLM with temperature > 0, so output changes every run.
# Use for: recommendations, creative text, explanations, opinions.
# ══════════════════════════════════════════════════════════════════
class ProbabilisticAgent:
    name = "ProbabilisticAgent"

    def run(self, query: str, temperature: float = 0.8) -> dict:
        # Probabilistic: same question → different answer each time
        answer = call_llm(
            prompt=query,
            model=CHAT_MODEL,
            temperature=temperature,
            system=(
                "You are a friendly tech sales assistant for a laptop store. "
                "Give concise, helpful answers. Max 3 sentences."
            ),
        )
        # Simulated confidence — lower for high temperature
        confidence = round(max(0.4, 1.0 - temperature * 0.5), 2)

        return {
            "agent": self.name,
            "type": "probabilistic",
            "answer": answer,
            "temperature": temperature,
            "confidence": confidence,  # approximate, not ground truth
        }


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — RAG (Retrieval-Augmented Generation)
# Step 1: retrieve relevant docs from dummy data (keyword match)
# Step 2: pass retrieved context + query to LLM
# More accurate than pure probabilistic because it's grounded.
# ══════════════════════════════════════════════════════════════════
class RAGAgent:
    name = "RAGAgent"

    def _retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """Simple keyword retrieval — in production use embeddings."""
        scored = []
        words = set(query.lower().split())
        for doc in FAQ_DOCS:
            topic_words = set(doc["topic"].split())
            text_words  = set(doc["text"].lower().split())
            score = len(words & topic_words) * 2 + len(words & text_words)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]

    def run(self, query: str) -> dict:
        retrieved = self._retrieve(query)

        if not retrieved:
            context = "No specific policy found."
        else:
            context = "\n".join(f"- {d['text']}" for d in retrieved)

        prompt = (
            f"Customer question: {query}\n\n"
            f"Relevant policies:\n{context}\n\n"
            "Answer clearly and concisely based only on the policies above."
        )
        answer = call_llm(
            prompt=prompt,
            model=CHAT_MODEL,
            temperature=0.3,   # low temp → more faithful to retrieved docs
            system="You are a helpful customer support agent.",
        )
        return {
            "agent": self.name,
            "type": "rag",
            "retrieved_docs": [d["topic"] for d in retrieved],
            "answer": answer,
            "confidence": 0.85 if retrieved else 0.3,
        }


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR — routes query to the right agent(s)
# Uses a small LLM call to classify intent, then dispatches.
# ══════════════════════════════════════════════════════════════════
class OrchestratorAgent:
    def __init__(self):
        self.deterministic = DeterministicAgent()
        self.probabilistic  = ProbabilisticAgent()
        self.rag            = RAGAgent()

    def _classify(self, query: str) -> str:
        """Ask LLM to classify query intent. Returns one of: data|policy|general"""
        prompt = (
            f"Classify this customer query into exactly one word.\n"
            f"Query: '{query}'\n\n"
            f"Categories:\n"
            f"- data     → pricing, inventory, sales numbers\n"
            f"- policy   → return, warranty, shipping, EMI, discount\n"
            f"- general  → recommendations, explanations, opinions\n\n"
            f"Reply with only one word: data, policy, or general."
        )
        result = call_llm(prompt, model=CHAT_MODEL, temperature=0.0)
        # Deterministic extraction — pick first recognized word
        for cat in ("data", "policy", "general"):
            if cat in result.lower():
                return cat
        return "general"   # safe fallback

    def run(self, query: str) -> dict:
        print(f"\n{Fore.CYAN}[Orchestrator] Classifying query...{Style.RESET_ALL}")
        intent = self._classify(query)
        print(f"{Fore.CYAN}[Orchestrator] Intent → {intent}{Style.RESET_ALL}")

        if intent == "data":
            result = self.deterministic.run(query)
            if result is None:
                result = self.probabilistic.run(query)   # fallback

        elif intent == "policy":
            result = self.rag.run(query)

        else:  # general
            result = self.probabilistic.run(query)

        result["intent_classified"] = intent
        return result


# ══════════════════════════════════════════════════════════════════
# INTERACTIVE CHAT LOOP
# ══════════════════════════════════════════════════════════════════
def print_result(result: dict):
    agent_colors = {
        "DeterministicAgent": Fore.GREEN,
        "ProbabilisticAgent": Fore.YELLOW,
        "RAGAgent":           Fore.BLUE,
    }
    color = agent_colors.get(result.get("agent", ""), Fore.WHITE)

    print(f"\n{color}{'─'*60}")
    print(f"Agent      : {result.get('agent', 'unknown')}")
    print(f"Type       : {result.get('type', '?')}")
    print(f"Intent     : {result.get('intent_classified', 'n/a')}")
    print(f"Confidence : {result.get('confidence', '?')}")
    if result.get("retrieved_docs"):
        print(f"Retrieved  : {result['retrieved_docs']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"{'─'*60}{Style.RESET_ALL}")


def main():
    print(f"{Fore.MAGENTA}")
    print("╔══════════════════════════════════════════╗")
    print("║  Multi-Agent Chat  (local Ollama models) ║")
    print("║  type 'quit' to exit                     ║")
    print("╚══════════════════════════════════════════╝")
    print(Style.RESET_ALL)

    orchestrator = OrchestratorAgent()

    demo_queries = [
        "What is the price of your laptops?",
        "Do you have EMI options?",
        "Which laptop would you recommend for a student?",
        "What are your sales numbers this quarter?",
        "Can I return a product if I don't like it?",
    ]

    print(f"{Fore.WHITE}Demo queries you can try:{Style.RESET_ALL}")
    for i, q in enumerate(demo_queries, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            query = input(f"{Fore.WHITE}You: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Allow typing a number to pick a demo query
        if query.isdigit() and 1 <= int(query) <= len(demo_queries):
            query = demo_queries[int(query) - 1]
            print(f"{Fore.WHITE}Using: {query}{Style.RESET_ALL}")

        result = orchestrator.run(query)
        print_result(result)


if __name__ == "__main__":
    main()
