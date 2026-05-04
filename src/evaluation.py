"""
RAG evaluation framework using RAGAS and custom metrics.

Measures:
- Faithfulness: Does the answer stick to the provided context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved chunks relevant?
- Context Recall: Did we retrieve all necessary information?
- Hallucination Rate: Custom metric for factual grounding
- Latency: End-to-end and per-stage timing
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset
from pydantic import BaseModel

from src.rag import query, RAGResult

logger = logging.getLogger(__name__)

EVAL_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval_results"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Test dataset ───────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "question": "What changes were made to Jinx in the most recent patches?",
        "ground_truth": "Jinx received adjustments to her base stats and abilities in recent patches.",
    },
    {
        "question": "Were there any changes to the dragon system?",
        "ground_truth": "The dragon system has seen changes including adjustments to dragon souls and elemental rift effects.",
    },
    {
        "question": "What items were reworked or added recently?",
        "ground_truth": "Several items have been reworked or added in recent patches, including changes to mythic items and their build paths.",
    },
    {
        "question": "What jungle changes were introduced?",
        "ground_truth": "Jungle changes have included adjustments to camp spawn timers, experience values, and jungle item modifications.",
    },
    {
        "question": "Were there any changes to the ranked system?",
        "ground_truth": "The ranked system has received updates including changes to LP gains, placement games, and split rewards.",
    },
    {
        "question": "What support champions received buffs?",
        "ground_truth": "Various support champions received buffs to improve their viability in the support role.",
    },
    {
        "question": "What nerfs were applied to top lane champions?",
        "ground_truth": "Several top lane champions received nerfs to balance their performance in solo queue and competitive play.",
    },
    {
        "question": "Were there any changes to ARAM?",
        "ground_truth": "ARAM has received balance changes including champion-specific adjustments and map modifications.",
    },
]


# ─── Custom metrics ─────────────────────────────────────────────────────────


class CustomMetrics(BaseModel):
    """Lightweight custom metrics that don't require an external LLM judge."""

    answer_length: int = 0
    context_utilization: float = 0.0  # What fraction of contexts were likely used
    response_confidence: float = 0.0  # Heuristic: absence of hedging language
    retrieval_mrr: float = 0.0  # Mean Reciprocal Rank based on similarity scores
    latency_retrieval_ms: float = 0.0
    latency_generation_ms: float = 0.0
    latency_total_ms: float = 0.0


def compute_custom_metrics(result: RAGResult) -> CustomMetrics:
    """Compute custom metrics from a RAG result."""
    answer = result.answer.lower()

    # Answer length
    answer_length = len(result.answer.split())

    # Context utilization: rough measure of how many context chunks
    # have terms appearing in the answer
    if result.contexts:
        used = 0
        for ctx in result.contexts:
            ctx_words = set(ctx.lower().split())
            answer_words = set(answer.split())
            overlap = len(ctx_words & answer_words)
            if overlap > 5:  # Meaningful overlap
                used += 1
        context_utilization = used / len(result.contexts)
    else:
        context_utilization = 0.0

    # Response confidence: lower if the answer contains hedging
    hedging_phrases = [
        "i don't have enough information",
        "i'm not sure",
        "i cannot",
        "i don't know",
        "unclear",
        "no relevant",
        "not mentioned",
        "not enough information",
    ]
    has_hedging = any(phrase in answer for phrase in hedging_phrases)
    response_confidence = 0.0 if has_hedging else 1.0

    # MRR from similarity scores
    if result.retrieval_scores:
        # Treat the first result above a threshold as the "relevant" one
        mrr = 0.0
        for i, score in enumerate(result.retrieval_scores):
            if score > 0.3:  # Similarity threshold
                mrr = 1.0 / (i + 1)
                break
        retrieval_mrr = mrr
    else:
        retrieval_mrr = 0.0

    return CustomMetrics(
        answer_length=answer_length,
        context_utilization=context_utilization,
        response_confidence=response_confidence,
        retrieval_mrr=retrieval_mrr,
        latency_retrieval_ms=result.retrieval_latency_seconds * 1000,
        latency_generation_ms=result.generation_latency_seconds * 1000,
        latency_total_ms=result.total_latency_seconds * 1000,
    )


# ─── RAGAS evaluation ───────────────────────────────────────────────────────


def build_ragas_dataset(results: list[RAGResult], ground_truths: list[str]) -> Dataset:
    """Convert RAG results into a HuggingFace Dataset for RAGAS."""
    data = {
        "question": [r.question for r in results],
        "answer": [r.answer for r in results],
        "contexts": [r.contexts for r in results],
        "ground_truth": ground_truths,
    }
    return Dataset.from_dict(data)


def run_ragas_evaluation(results: list[RAGResult], ground_truths: list[str]) -> dict:
    """
    Run RAGAS evaluation metrics.

    Metrics computed:
    - faithfulness: Is the answer faithful to the context?
    - answer_relevancy: Is the answer relevant to the question?
    - context_precision: Are retrieved docs relevant?
    - context_recall: Did we retrieve everything needed?

    Note: RAGAS uses an LLM as a judge. By default it uses OpenAI,
    but we configure it to use a local HuggingFace model.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

        logger.info("Setting up RAGAS with local HuggingFace models...")

        # Use the same models we already have loaded for judging
        llm = HuggingFacePipeline.from_model_id(
            model_id=config.model.llm_model,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 256,
                "temperature": 0.1,
            },
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=config.model.embedding_model,
        )

        wrapped_llm = LangchainLLMWrapper(llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

        dataset = build_ragas_dataset(results, ground_truths)

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_embeddings,
        )

        return dict(ragas_result)

    except ImportError as e:
        logger.warning(f"RAGAS dependencies not fully installed: {e}")
        logger.info("Falling back to custom-only evaluation")
        return {"error": str(e)}
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {"error": str(e)}


# ─── Full evaluation pipeline ───────────────────────────────────────────────


class EvaluationReport(BaseModel):
    timestamp: str
    num_questions: int
    ragas_scores: dict = {}
    custom_metrics_avg: dict = {}
    per_question: list[dict] = []


def run_full_evaluation(
    questions: list[dict] | None = None,
    save_report: bool = True,
) -> EvaluationReport:
    """
    Run the complete evaluation pipeline.

    1. Run each question through the RAG pipeline
    2. Compute custom metrics per question
    3. Run RAGAS evaluation for LLM-judged metrics
    4. Aggregate and save report
    """
    if questions is None:
        questions = EVAL_QUESTIONS

    logger.info(f"Running evaluation on {len(questions)} questions")

    # Step 1: Run queries
    rag_results = []
    custom_metrics_list = []
    per_question_data = []

    for q_data in questions:
        question = q_data["question"]
        logger.info(f"Evaluating: {question}")

        result = query(question)
        rag_results.append(result)

        # Custom metrics
        custom = compute_custom_metrics(result)
        custom_metrics_list.append(custom)

        per_question_data.append({
            "question": question,
            "answer": result.answer,
            "ground_truth": q_data["ground_truth"],
            "num_contexts": len(result.contexts),
            "source_patches": result.source_patches,
            "top_similarity": result.retrieval_scores[0] if result.retrieval_scores else 0,
            "custom_metrics": custom.model_dump(),
        })

    # Step 2: Aggregate custom metrics
    avg_custom = {}
    if custom_metrics_list:
        metric_fields = [
            "answer_length", "context_utilization", "response_confidence",
            "retrieval_mrr", "latency_retrieval_ms", "latency_generation_ms",
            "latency_total_ms",
        ]
        for field_name in metric_fields:
            values = [getattr(m, field_name) for m in custom_metrics_list]
            avg_custom[field_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    # Step 3: RAGAS evaluation
    ground_truths = [q["ground_truth"] for q in questions]
    ragas_scores = run_ragas_evaluation(rag_results, ground_truths)

    # Step 4: Build report
    report = EvaluationReport(
        timestamp=datetime.now().isoformat(),
        num_questions=len(questions),
        ragas_scores=ragas_scores,
        custom_metrics_avg=avg_custom,
        per_question=per_question_data,
    )

    if save_report:
        report_path = EVAL_OUTPUT_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        logger.info(f"Evaluation report saved to {report_path}")

    return report


def print_report(report: EvaluationReport):
    """Pretty-print an evaluation report to the console."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Header
    console.print(Panel(
        f"[bold]RAG Evaluation Report[/bold]\n"
        f"Timestamp: {report.timestamp}\n"
        f"Questions evaluated: {report.num_questions}",
        title="LoL Patch Notes RAG",
    ))

    # RAGAS scores
    if report.ragas_scores and "error" not in report.ragas_scores:
        table = Table(title="RAGAS Metrics (LLM-judged)")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right", style="green")

        for metric, score in report.ragas_scores.items():
            if isinstance(score, (int, float)):
                table.add_row(metric, f"{score:.4f}")

        console.print(table)

    # Custom metrics
    if report.custom_metrics_avg:
        table = Table(title="Custom Metrics (Aggregated)")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for metric, stats in report.custom_metrics_avg.items():
            table.add_row(
                metric,
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
            )

        console.print(table)

    # Per-question results
    table = Table(title="Per-Question Results")
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", max_width=40)
    table.add_column("Answer Preview", max_width=40)
    table.add_column("Chunks", justify="right")
    table.add_column("Top Sim", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for i, q in enumerate(report.per_question, 1):
        answer_preview = q["answer"][:60] + "..." if len(q["answer"]) > 60 else q["answer"]
        table.add_row(
            str(i),
            q["question"],
            answer_preview,
            str(q["num_contexts"]),
            f"{q['top_similarity']:.3f}",
            f"{q['custom_metrics']['latency_total_ms']:.0f}",
        )

    console.print(table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.config import config
    report = run_full_evaluation()
    print_report(report)
