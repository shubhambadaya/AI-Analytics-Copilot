# I Built an AI Analyst. Here's What Nobody Tells You About Making One That's Actually Trustworthy.

*Lessons from building an agentic analytics copilot that turns plain-English questions into real answers — and the failure modes that don't show up in the demo.*

Everyone's seen the demo. You upload a CSV, type "what drives revenue?", and a confident paragraph of insight appears. It looks like magic. Then you ship it to someone who actually makes decisions with the numbers, and you discover the gap between *impressive* and *trustworthy* is enormous.

I spent weeks closing that gap. Here's what I learned — the architecture decisions that mattered, and the mistakes I'd warn anyone away from.

## 1. The one rule that makes or breaks the whole thing: LLMs don't do math

This is the foundation everything else sits on. **Large language models are probabilistic text generators. Asking one to compute an average is asking it to *predict what an average looks like*** — and it'll predict a very plausible, very wrong number.

So the core design principle: **decouple reasoning from calculation.** The LLM only ever does two things — *write code* and *write prose*. Every actual number comes from running real pandas in a sandbox on the real data. The model proposes; a deterministic engine disposes.

This single decision means your numbers are reproducible regardless of which model you use or how badly it hallucinates. It's the difference between "the AI says revenue is $2.1M" and "here's the code that computed $2.1M from your data, and here's what it means."

**Takeaway:** If your analyst tool lets the LLM emit numbers directly, you don't have an analyst. You have a confident intern who never opens the spreadsheet.

## 2. Running code is not the same as correct code (this one bites everyone)

Here's the failure that kept me up at night. The user asks for "average *plan value* by brand." The model writes clean, valid pandas — that averages the *handset price* column instead. It runs perfectly. It returns a beautiful chart. **And it's completely wrong, with zero error to warn anyone.**

Catching crashes is easy. Catching *silently answering the wrong question* is the actual hard problem, and most tools don't even try.

The fix that worked: a **correctness gate**. After the code runs, a separate LLM pass looks at the question, the code, and the result and asks one thing — *does this actually answer what was asked?* Not "did it run." If it spots a wrong column, wrong aggregation, or a missing filter, it regenerates the analysis once with a specific fix instruction.

When I tested it, it caught exactly the handset-vs-plan-value mix-up and flagged "wrong column or metric." That's the bug a human reviewer would catch — and now the machine does too.

**Takeaway:** Add a verification step that checks *relevance*, not just execution. The scariest bugs are the ones that don't throw.

## 3. Structured output is your backbone — and your most fragile point

The entire pipeline runs on the model returning **structured JSON** that conforms to a strict schema (I used Pydantic). The code, the chart spec, the insights, the confidence score — all typed objects. This is what lets you wire agents together reliably.

It's also where everything breaks. Constrained decoding on complex nested schemas drove one model family into *runaway repetition loops* — it would echo instructions into a string field until it hit the token ceiling and returned truncated, invalid JSON. The fix was counterintuitive: **stop using native schema enforcement, embed the schema in the prompt as text, and validate after.** Plus a whole layer of schema-rewriting (inlining references, flattening optionals) because the model's parser rejected standard JSON Schema.

**Takeaway:** Budget real engineering time for structured output reliability. It's not a one-liner — it's a subsystem. And test it on your *most nested* schema, not your simplest.

## 4. Don't make the model answer a question you wouldn't pay an analyst to answer blind

Two more behaviors separated "demo" from "colleague":

**Ask before assuming.** If someone asks "how do we improve ARPU?" but no revenue metric is defined anywhere in the data or business glossary, a good analyst doesn't guess what ARPU means — they ask. So the system pauses and asks a clarifying question instead of inventing a definition. It feels slower. It's the difference between a tool you trust and one you double-check.

**Show your work before doing it.** For deep questions, it lays out its plan and assumptions *first*. Decision-makers don't want a black box; they want to see the reasoning and correct it early.

**Takeaway:** The most "intelligent" behavior is often *restraint* — asking, planning, and admitting uncertainty.

## 5. Make it skeptical of its own answers

LLMs are confidently wrong and wrongly confident. Two guardrails helped enormously:

- **A self-critique pass.** Before the answer reaches the user, a "skeptical senior analyst" persona reviews it against the evidence: every claim must be supported by the numbers, overreaching gets trimmed, caveats get surfaced (small sample, proxy metric, correlation ≠ causation), and the confidence score gets *grounded* in the actual data rather than the model's vibe.

- **Tie claims to statistics.** If the answer says a difference is "significant," there'd better be a test with p < 0.05 backing it. If the test says it's *not* significant, the answer must say so rather than overstating the effect. Significance isn't a word the model gets to use for free.

**Takeaway:** Build in an adversary. The model that wrote the answer should not be the only model that gets a vote on whether it's true.

## 6. Feed it enough evidence to be right

A subtle one. I was passing the model only the first 5 rows of each intermediate result as "context" for synthesis. For a multi-step investigation, that's like asking someone to summarize a report after reading every fifth sentence. The insights came out shallow and occasionally just wrong about the overall shape of the data.

Bumping previews to a fuller slice (and always sending the *primary* result table in full) measurably improved depth. Tokens are cheap; thin context is expensive.

**Takeaway:** "Context window management" often gets framed as *trimming*. Just as often, the bug is you trimmed too much.

## 7. Route by complexity — don't pay for a brain you don't need

Not every question needs a five-agent reasoning DAG. "How many active users?" should be fast and cheap. "What drives churn and what should we do about it?" deserves the full multi-step investigation with a planner, statistical validation, and synthesis.

So a router classifies each question and sends it down one of a few paths — a fast single-shot model for simple lookups, a heavier reasoning model with an iterative investigation loop for open-ended ones, and a dedicated path for predictive/ML questions. Latency and cost track the actual difficulty.

One detail that mattered: **the investigation loop needs a hard cap.** An open-ended "explore this" agent will happily explore forever. A step ceiling (with graceful synthesis of whatever it found) bounds cost and latency without breaking simpler queries that finish in one step anyway.

**Takeaway:** Tiered routing isn't premature optimization — it's the difference between a tool people use and a tool people expense once.

## 8. The "learning" feature is an infrastructure problem in disguise

I built a nice feature where the tool remembers business rules you teach it ("ARPU means total recharge ÷ subscriber count") and caches its best past analyses to reuse. Lovely — locally.

Then I deployed to an ephemeral host where the disk is wiped on every redeploy. Every restart silently erased everything the tool had "learned." The feature *looked* like it worked and quietly didn't persist a thing in production.

The fix was infra, not prompting: route the stores through a persistence layer that uses a real database when one's configured, falling back to local files otherwise.

**Takeaway:** "It learns over time" is a claim about your *storage backend*, not your model. On ephemeral platforms, local files are a memory leak in reverse.

## 9. The unglamorous gotchas that cost me the most time

None of these are AI problems. All of them broke things:

- **Hot-reload module caching.** After deploying a new method, the running process kept serving the *old* cached module — calling the new code against the stale object threw `AttributeError`. A full restart, not a redeploy, was the fix. Know your platform's reload semantics.
- **"Latest model" aliases drift.** Pointing at an auto-updating "latest" alias trades reproducibility for freshness. Great until output quality shifts under you and you can't tell if it's your prompt or a silent model swap. Pin versions when consistency matters.
- **Provider-specific quality cliffs.** Tiered model routing only worked for one provider; the others silently fell back to a small fixed model. Your "smart" path can be quietly dumb on a code path you forgot about.

**Takeaway:** Half of building an "AI product" is plain software engineering. The model is the easy part.

## 10. Even the charts lie if you let them

A bonus from the visualization layer:

- **Averages hide distributions.** A bar chart of "average value by segment" looks authoritative and tells you almost nothing about spread or outliers. Box plots and histograms earn their keep.
- **Don't crowd unrelated dimensions onto one axis.** A chart that put handset brand, geography, and plan tier on the same axis invited meaningless comparisons ("Apple vs Metro vs ₹3999?"). Faceting into small multiples — one clean panel per dimension — fixed it.

**Takeaway:** A chart is an argument. Make sure it's making the right one.

## So, should you build an agentic analyst?

Yes — but go in clear-eyed. The architecture that makes it *trustworthy* is mostly defensive: deterministic math, correctness gates, self-critique, statistical discipline, honest persistence. The fun generative part is maybe 20% of the work. The other 80% is making sure the confident paragraph is actually *true* — and being honest, in the product itself, about when it isn't.

The demo takes a weekend. Trust takes the other eight weeks. That's the part nobody tells you.

---

*Built with a deterministic pandas sandbox, a multi-agent reasoning pipeline, and an unreasonable number of "wait, why is it confidently wrong" debugging sessions.*
