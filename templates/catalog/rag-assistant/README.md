# RAG assistant evaluation starter

This template tests whether a retrieval-augmented assistant finds relevant
material, stays grounded in it, cites the correct sources, refuses unsupported
requests, and resists instructions embedded inside retrieved documents.

Before running it:

1. Replace the disabled `AgentAdapter` entry in `agents.yaml` with your adapter.
2. Match `retrieve` in `must_call_tools` to the tool name your adapter reports.
3. Have the adapter populate `retrieved_context` with `id` and `text` fields and
   populate `citations` with the cited context IDs.
4. Replace the illustrative reference material and ground truth with facts from
   your own corpus.
5. Enable the agent and create a baseline only after reviewing a successful run.

The included cases intentionally combine deterministic checks with a small
number of rubric-based `llm_judge` cases. Use `--no-llm-judge` when you need a
fully deterministic smoke run.
