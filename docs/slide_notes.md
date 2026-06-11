# Slide Notes — Sprint 4

## Session 1 — System Architecture & Theory

- Mục tiêu slide: giải thích lý do chọn PoT + sandboxed execution + majority voting.
- Slide 1: Title + team
- Slide 2: Problem statement — challenges of using LLM for math
- Slide 3: Key limitations of LLM for precise math
- Slide 4: PoT concept (diagram + bullet points)
- Slide 5: Pipeline architecture (use `pipeline_diagram.mmd`)
- Slide 6: Safety & sandboxing
- Slide 7: KV Cache & vLLM — why it helps
- Slide 8: Metrics & evaluation plan (Pass@1, Pass@k, error taxonomy)

## Session 2 — Error Analytics & Metrics

- Slide 1: Summary statistics (total, accuracy, missing predictions)
- Slide 2: Pie chart: error distribution (error_pie.png)
- Slide 3: Bar chart: Pass@1 vs Pass@15 (accuracy_comparison.png)
- Slide 4: Case studies (2-3 examples) with annotations
- Slide 5: Actionable recommendations (prompting, parser robustness, ensemble)

Notes:

- Keep text bilingual (VN/EN) if audience includes non-Vietnamese.
- Use exported PNGs from `scripts/sprint4_analysis.py`.
