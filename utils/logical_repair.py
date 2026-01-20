"""
run the logical repair first (so you get concrete repair plans / conflict explanations), then use an LLM to choose between alternative repair plans based on semantic priors, provenance, and other non-logical signals.
"""

# run logmap repair
# java -jar logmap-matcher.jar --repair existing_mappings.rdf SOURCE.owl TARGET.owl


def repair_plan_prompt(o1, o2, repair_plans):
    prefix_prompt = f"""
You are an expert in ontology alignment.
We have a pair of ontologies {o1} and {o2}. A logical repair algorithm produced several repair plans that remove different mappings to restore coherence.

Rules for ranking (in order of importance):

1) Plan must **preserve coherence** (we will verify with a reasoner). If a plan would reintroduce obvious contradictions prefer otherwise.
2) Prefer plans that remove fewer mappings.
3) Prefer plans that keep mappings with higher original confidence.
4) Prefer plans that keep mappings whose labels and definitions strongly match.

Candidate metadata (below) — return JSON array sorted by score.
"""
    
    repair_plans_prompt = []

    for plan in repair_plans:
        plan_prompt = f"""
- "plan_id": {plan['plan_id']}
- "score": {plan['score']}
- "reason": {plan['reason']}
"""
        repair_plans_prompt.append(plan_prompt)

    return prefix_prompt + "\n".join(repair_plans_prompt) + "\n"