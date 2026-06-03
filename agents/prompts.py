EXTRACTOR_PROMPT = """\
You are an expert conversation intelligence extractor.

You receive a conversation between a user and an assistant.

Your task is to extract ONLY USER INFORMATION and organize it into TWO distinct layers:

------------------------------------------------------------
1. KNOWLEDGE LAYER — what the user knows, cares about, or works on
------------------------------------------------------------
Extract:
- macro-topics and domains
- goals and objectives
- constraints and limitations
- stable preferences and context

This is long-term memory. Each item should be a self-contained, reusable unit.

------------------------------------------------------------
2. BEHAVIOR FEEDBACK LAYER — how the assistant should behave
------------------------------------------------------------
Extract ONLY signals about interaction quality:
- what the user considered correct or incorrect
- complaints about retrieval, reasoning, verbosity, or structure
- explicit or implicit requests for behavior changes
- evaluation signals (positive or negative)

This is NOT knowledge. It is policy guidance for future interactions.

------------------------------------------------------------

Rules:
1. Ignore all assistant messages — extract only from user turns
2. Do NOT over-fragment: merge related concepts into macro units
3. Do not infer intentions, emotions, or beliefs not explicitly stated
4. Keep outputs compact but information-dense
5. signals = direct textual evidence from the user's words (short phrases or keywords)

------------------------------------------------------------

Output format (STRICT JSON, no markdown):

{{
  "knowledge": [
    {{
      "topic": "short label for the macro-topic",
      "summary": "one or two sentences describing what the user said",
      "signals": ["direct phrase or keyword from user", "..."]
    }}
  ],
  "behavior_feedback": [
    {{
      "signal_type": "retrieval_error | preference | complaint | evaluation | constraint",
      "description": "what the user said or implied about interaction quality",
      "severity": "low | medium | high"
    }}
  ]
}}

If nothing exists in a section, return an empty array for that key.
"""

MERGER_PROMPT = """\
You are an agent that manages a structured knowledge base organized as a DAG.

You receive:
- A CHUNK: a new piece of information to integrate
- CANDIDATE NODES: existing nodes that may be related

Each node has an implicit level in the graph:
- ROOT  = macro-topic (broad domain, stable)
- MID   = concept (intermediate abstraction)
- LEAF  = detail (specific, frequently updated)

------------------------------------------------------------
Actions — choose exactly ONE:

UPDATE  → The chunk refines, extends, or corrects an existing node.
          Use when the chunk and the target node share the same concept
          and the new information adds detail without changing scope.

MERGE   → The chunk and an existing node are semantically equivalent
          or heavily overlapping. Collapse them into a single node.
          Use sparingly — only when similarity is extremely high.

ADD     → No candidate is close enough. The chunk introduces a concept
          not yet represented in the graph. Create a new node.

SKIP    → The chunk adds no new information. The existing node already
          covers it completely.

------------------------------------------------------------
Hierarchy rules:
- Do NOT collapse nodes across levels unless similarity is near-identical
- ROOT nodes must remain stable — prefer UPDATE over MERGE at root level
- LEAF nodes may evolve more freely

------------------------------------------------------------
Output (strict JSON, no markdown):

{{
  "action": "update | merge | add | skip",
  "target_node_id": "id of the target node, or null if action is add",
  "new_content": "full updated content for the node, or null if action is skip",
  "new_summary": "updated navigational summary, or null if action is skip",
  "rationale": "one sentence explaining the decision"
}}
"""

SUMMARY_UPDATER_PROMPT = """\
You maintain a navigational summary for a node in a knowledge DAG.

The summary answers one question:
"If I am searching for a concept, should I descend into this subtree?"

You receive the node's current content and its children's summaries.

Rules:
- Describe what exists in this subtree, not what the node itself is
- Emphasize discrimination: what is here vs what is NOT here
- Write in plain prose, no bullet points
- Maximum 3 sentences

Output ONLY the summary text, nothing else.
"""

QUERY_BUILDER_PROMPT = """\
You are a search assistant. Your job is to transform a conversation into a single semantic search query.

You receive the last N turns of a conversation between a user and an assistant.

Produce ONE plain-text query in Italian that captures:
- The user's primary intent in the most recent turn
- Relevant context from previous turns (entities, topic continuations)
- Key concepts or named entities that should anchor the search

Rules:
- Output a single sentence or short phrase — no lists, no labels, no explanation
- If the conversation contains multiple intents, focus on the most recent one
- If the context is ambiguous, default to the most specific interpretation

Output ONLY the query. No preamble, no punctuation other than what the query itself needs.
"""

RETRIEVAL_GATE_PROMPT = """\
You are a routing filter. Decide whether a conversation requires searching a knowledge base.

Answer YES if the last user message:
- asks a factual or domain-specific question
- references a topic, entity, or concept that may require context
- continues a technical or informational thread from previous turns

Answer NO if the last user message:
- is a greeting, farewell, or social exchange (e.g. "ciao", "come stai", "grazie")
- is a simple acknowledgment or filler ("ok", "capito", "sì")
- contains no informational intent

Output ONLY the word YES or NO. Nothing else.
"""

RELEVANCE_FILTER = """\
You are a relevance filter for a semantic retrieval system.

Query:
{query_text}

Candidate nodes:
{nodes_text}

A node is relevant if it directly addresses the query or contains information
that would meaningfully help answer it. Exclude nodes that are only tangentially related.

Return ONLY a JSON array of relevant node IDs.
If no nodes are relevant, return [].
"""

MERGE_DECISION_PROMPT = """\
You are a knowledge graph integration agent.

You receive:
- NEW CHUNKS: new information units extracted from a recent conversation
- EXISTING NODES: nodes currently in the knowledge DAG that are relevant to these chunks

Your task is to decide how to integrate each chunk (or part of it) into the graph.

------------------------------------------------------------
For each chunk, produce ONE OR MORE decisions.
A chunk can be split if different parts belong to different nodes.

Available actions:

ADD_ROOT  → The chunk introduces a completely new macro-topic with no
            relation to existing nodes. Create a new root node.

ADD_CHILD → The chunk introduces a new concept that belongs under an
            existing node. Specify the parent node from EXISTING NODES.

UPDATE    → The chunk refines, extends, or corrects an existing node.
            The concept is the same, the information is new.
            Provide the full updated content.

MERGE     → The chunk is semantically equivalent to an existing node.
            Fold it in. Use sparingly — only when overlap is near-total.
            Provide the full merged content.

SKIP      → The chunk adds no new information. An existing node already
            covers it completely.

------------------------------------------------------------
Hierarchy rules:
- Do NOT collapse nodes across levels unless overlap is near-identical
- ROOT nodes are stable: prefer ADD_CHILD or UPDATE over ADD_ROOT
- LEAF nodes may evolve freely

------------------------------------------------------------
Output (strict JSON, no markdown):

{
  "decisions": [
    {
      "chunk_id": "id of the chunk being decided",
      "action": "add_root | add_child | update | merge | skip",
      "target_node_id": "id of the existing node to update/merge/skip, or null",
      "parent_node_id": "id of the parent node if action is add_child, or null",
      "new_content": "full new content for the node, or null if action is skip",
      "new_summary": "updated navigational summary, or null if action is skip",
      "rationale": "one sentence explaining this decision"
    }
  ]
}

If a chunk is split across multiple nodes, emit one decision object per fragment.
"""