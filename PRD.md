# Product Requirements Document: Service Provider Agent

**Status:** Initial draft · **Date:** 2026-10-08
**Repository:** `shahintaesheikh/service-provider-agent`
**Delivery mode:** `direct-PR` (no validation pipeline; PR-based workflow)

---

## 1. Product Vision

A conversational agent that helps Santa Barbara homeowners find the right home service provider. A user describes a home problem in natural language ("Water started coming into my basement last night after the storm. I don't know who to call.") and the agent converses, classifies the issue, and generates a dispatchable lead matched to a real, verifiable local provider.

## 2. Key Outcomes

| Outcome | Definition | Measurement |
|---------|-----------|-------------|
| **Conversion Rate** | How many conversations result in a dispatchable lead | Trainer log: leads generated / conversations completed |
| **Lead Quality** | Would a real service provider take this lead? | Jev Noul calibrated probability + human feedback loop via trainer log |

## 3. User Flow

```
User describes home problem in natural language
  → Agent extracts intent (NLU / ChatOpenAI)
  → Agent clarifies if vague (deterministic heuristics + multi-turn loop)
  → Agent classifies problem category + subcategory (Jev Choice)
  → Agent scores urgency dimensions (Jev Score + Noul → code composite)
  → Agent selects provider APIs to query (Jev Choice)
  → Code queries Google Places + Yelp Fusion APIs
  → Code merges, deduplicates, ranks providers
  → HITL Validator: auto-route or escalate to human
  → Agent checks lead quality (Jev Noul)
  → Trainer log records full session + outcome
```

### Example Conversation

> **User:** Water started coming into my basement last night after the storm. I don't know who to call.
>
> **Agent:** I can help! A couple questions — where in Santa Barbara is the property?
>
> **User:** Corner of Chapala and Victoria, downtown.
>
> **Agent:** Is water still actively coming in, or has it slowed?
>
> **User:** It's slowed to a trickle. About an inch standing.
>
> **Agent (after processing):** ✅ Lead ready. Matched to Lewis Plumbing (A+ BBB, 4.6★, CSLB licensed, open now). Here's the summary...

## 4. Location Scope

**Santa Barbara, CA** (city proper, Goleta, Carpinteria, Montecito). Provider radius extends to broader Santa Barbara County for specialty trades.

## 5. Architecture

### Pipeline

```
User Input
  │
  ▼
LangGraph (orchestration + state management)
  │
  ├── ChatOpenAI → NLU / Entity Extraction
  │
  ├── Jev Choice → Category Classification (10 home service categories)
  ├── Jev Choice → Subcategory Classification (3–6 per category)
  ├── CODE → Over-escalation Trap Patterns (keyword-based false alarm guard)
  │
  ├── Jev Choice → API Selection (Google Places / Yelp / both / none)
  ├── CODE → Google Places API call
  ├── CODE → Yelp Fusion API call
  ├── CODE → Merge, deduplicate, rank results
  │
  ├── Jev Score + Noul → Urgency Dimension Signals
  ├── CODE → Composite Urgency Score (weighted blend of Jev signals + safety indicators)
  ├── CODE → HITL Threshold = composite urgency × P(error) × Cost(error)
  │
  ├── HITL Validator → Auto-route or pause for human review
  ├── HITL Override → Apply human corrections
  ├── HITL Trainer Log → Full record with provenance
  │
  ├── Jev Noul → Lead Quality Check
  └── CODE + DB → Lead persistence + analytics
```

### Key Technology Decisions

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Orchestration | LangGraph (Python) | Proven DAG pattern for agent pipelines |
| NLU & Entity Extraction | ChatOpenAI (GPT-4.1-mini) | Free-form text extraction, multi-turn state management |
| Classification | Jev Choice (TypeSafe System One) | Typed, calibrated, non-hallucinating, ~$0.0001/call |
| API Selection Decisions | Jev Choice | Which provider API to query based on problem context |
| Urgency Signals | Jev Score + Noul | Dimension-level inputs (water spread, safety hazard) → code calculates composite |
| Urgency Composite | Code calculation | Jev feeds signals; code blends them with over-escalation traps for the final HITL threshold |
| Provider Matching | Google Places API + Yelp Fusion API | Live, real, verifiable provider data |
| License Verification | CA CSLB (Contractors State License Board) | Free, essential for regulated trades |
| Lead Quality | Jev Noul | Calibrated probability of conversion |
| HITL + Logging | Reference project's trinity | Validator → Override → Trainer, kept verbatim |
| Voice / Chat UI | iPhone-style shell (from reference project) | ElevenLabs STT/TTS (optional) + text composer fallback |

### Jev Integration Detail

| Component | Primitive | Options / Range | Confidence Threshold |
|-----------|-----------|-----------------|---------------------|
| Category Classification | Choice | Plumbing, HVAC, Electrical, Roofing, Water Damage, Appliance Repair, Pest Control, Locksmith, Handyman, Not applicable | ≥0.8 auto, 0.4–0.8 review, <0.4 escalate |
| Subcategory Classification | Choice | 3–6 per category (e.g., pipe_leak, drain_clog, water_heater, toilet_repair, faucet_repair) | ≥0.8 auto |
| API Selection | Choice | Google Places, Yelp, Both, None | ≥0.7 auto |
| Urgency Factor: water spreading? | Score | 0–1 | Raw signal (code weights it) |
| Urgency Factor: safety hazard? | Noul | Yes / No | Raw signal (code weights it) |
| Urgency Factor: contained? | Noul | Yes / No | Raw signal (code weights it) |
| Lead Quality | Noul | Likely to convert / Not likely | ≥0.8 auto-accept, <0.8 routes to review |

**Cardinal rule:** Jev does NOT make the composite urgency decision. It supplies calibrated dimension signals; code calculates the composite urgency score, which feeds the HITL threshold (derived from the reference project's proven Risk = P × C formula). This is a System 2 task, and Jev is purpose-built for System 1 judgments.

## 6. HITL Trinity (from Reference Project)

The proven Validator → Override → Trainer pattern is carried forward **verbatim**:

| Node | Function | Details |
|------|----------|---------|
| **Validator Gate** | Checks `needs_human_review`. If false → auto-routes. If true → pauses via `interrupt()` for a human reviewer. | Routing decision logged for audit trail. |
| **Override Node** | Applies human corrections (category, subcategory, urgency) to the AI prediction before final logging. | Corrections recorded in trainer log. |
| **Trainer Log** | Records full transcript + AI prediction (with provenance: original category, urgency dimensions, Jev confidence, HITL trigger reason) + human override (if any) + final decision. | Powers both conversion rate and lead quality metrics. MVP writes to local JSON files (`data/leads/`); production will use a proper database (Postgres/Supabase). |

### HITL Threshold Calculation

```
Composite Urgency = w1 × jev_water_spread_score
                  + w2 × jev_safety_hazard_noul
                  + w3 × jev_contained_noul_inverse
                  + w4 × keyword_safety_signals (sparking, gas, smoke, etc.)

HITL Threshold = Composite Urgency × P(error) × Cost(error)

P(error) = 0.40 × (1 - confidence) + 0.30 × disagreement + 0.30 × historical_rate
Cost(error) = 0.50 × severity_weight + 0.30 × scope_factor + 0.20 × context_modifier

If HITL Threshold ≥ 0.30 → needs_human_review = True
```

## 7. Provider Data Strategy

### Source Stack

| Tier | Source | Purpose | Cost |
|------|--------|---------|------|
| **1** | Google Places API (New) | Primary provider search by category + location. Returns name, phone, address, hours, rating, website. | ~$0–6/mo for Santa Barbara MVP |
| **2** | Yelp Fusion API | Cross-reference reviews + ratings. Free tier. | Free (rate-limited) |
| **3** | CA Contractors State License Board (CSLB) | License verification for plumbers, electricians, HVAC, roofers. Free. | Free |
| **4** | Better Business Bureau (BBB) | Trust signal (no official API). Used for verification layer. | Free (scraping) |

### Santa Barbara Provider Landscape

95+ real, BBB-verified providers surveyed across 9 categories:

| Category | Providers | Example (Top Pick) |
|----------|-----------|-------------------|
| Plumbing | 15 | Lewis Plumbing · (805) 569-1060 · A+ BBB |
| HVAC | 6 | Crocker Refrigeration · (805) 965-6326 · A+ BBB |
| Electrical | 6 | Brady Electric · (805) 452-1606 · A+ BBB |
| Roofing | 6 | Quality Roofing of Santa Barbara · (805) 965-2416 · A+ BBB |
| Water Damage | 5 | Servpro of Santa Barbara · (805) 963-0606 · A+ BBB |
| Appliance Repair | 4 | The Appliance Store · (805) 963-3023 · A+ BBB |
| Pest Control | 6 | Santa Barbara Pest Control · (805) 563-8888 · A+ BBB |
| Locksmith | 4 | Tri-County Locksmiths · (805) 967-4300 · A+ BBB |
| Handyman | 5 | A Jack of All Trades · (805) 708-5466 · A+ BBB |

### Provider Matching Cascade

```
subcategory → derive search terms (pipe_leak → "plumber", "plumbing repair")
  → Google Places Text Search (Santa Barbara, CA)
  → Yelp Fusion search (cross-reference)
  → Merge + deduplicate by phone/name
  → Filter by: open now, distance < 50mi, rating ≥ 3.0
  → Sort by: rating desc, review count desc
  → Return top 3 candidates
  → CSLB lookup for each candidate (regulated trades only)
  → Attach license status badge
```

## 8. UX Prototype

Based on the reference project's iPhone-style voice dispatch shell:

- **Phone mockup:** Status bar → avatar + agent name → chat transcript (iMessage-style bubbles) → call controls (mic, speaker, reset) → text composer
- **Side panel:** Lead result with provider match cards showing name, phone, address, rating badges (BBB, Google, CSLB), open status, distance
- **Voice option:** ElevenLabs Scribe v2 STT + Flash v2.5 TTS (optional, text always available as fallback)

## 9. Cost Model

| Component | LLM-only | Jev Hybrid |
|-----------|----------|------------|
| NLU / Entity Extraction | $10–20 | $10–20 (unchanged) |
| Classification | $10–20 | **~$0.50** |
| Urgency Dimensions | $10–20 | **~$0.50** |
| API Selection | $5–10 | **~$0.50** |
| Lead Quality | $5–10 | **~$0.25** |
| **Total (5K sessions/mo)** | **$40–80** | **~$12–22** |

Jev input cost: $0.042/MTok. No output token cost. Sub-500ms per call.

## 10. Reference Projects

| Project | Role | Location |
|---------|------|----------|
| `rich-shahin-finalproject` | Source architecture (LangGraph DAG, HITL trinity, prompt caching, clarification heuristics, over-escalation traps, vendor cascade, trainer log, ElevenLabs UI) | `projects/rich-shahin-finalproject/` (cloned from `acm-industry/rich_shahin_finalproject`) |

### Architecture Transfer Summary

| Pattern | Transfer | Carried forward? |
|---------|----------|------------------|
| LangGraph DAG wiring | node/edge/state pattern | ✅ As-is |
| State schema | TypedDict + Pydantic models | ✅ Adapted fields |
| Clarification heuristics | 4 deterministic gates | ✅ Adapted to home services |
| Multi-turn loop | interrupt() pattern | ✅ As-is |
| RAG pipeline | over-retrieve + rerank + consensus | ✅ As-is (may defer without corpus) |
| Prompt caching | static system + dynamic user split | ✅ As-is |
| Over-escalation traps | keyword pattern → correction | ✅ Adapted to home scenarios |
| Vendor cascade | specialty → location → rating | 🔶 Adapted: static JSON → live API |
| HITL trinity | Validator → Override → Trainer | ✅ As-is (verbatim) |
| Trainer log | full transcript + provenance | ✅ As-is |
| Voice UI | ElevenLabs Scribe + Flash | ✅ As-is (optional) |
| CBRE-specific data | buildings, vendors, profiles | ❌ Replaced entirely |

## 11. Decisions Made

| Question | Decision |
|----------|----------|
| Jev for urgency scoring? | **No.** Jev provides dimension signals only. Composite urgency is a code calculation, factored into the HITL threshold formula. |
| LLM-first fallback while waiting for Jev access? | **No.** We have Jev access. Build Jev-first. |
| HITL pattern to use? | **Reference project's trinity.** Validator → Override → Trainer, kept verbatim. |
| Data privacy concern with Jev API? | **Not a concern.** |
| Santa Barbara for MVP? | **Yes.** 95+ BBB-verified providers across 9 categories. |
| TypeSafe skill installed? | **Yes.** At `.agents/skills/typesafe-ai/`. |
| Delivery mode? | **direct-PR.** No no-mistakes pipeline. Early product — ship and demo first. |
| Repository? | **GitHub private.** `shahintaesheikh/service-provider-agent`. |

## 12. Open Questions

*None currently resolved — the captain has addressed all scout open questions.*

---

*This PRD was compiled from three scout reports:*
1. *Santa Barbara provider survey (`data/home-service-agent-survey-real-service-p-ff/report.md`)*
2. *Architecture transfer analysis (`data/rich-shahin-finalproject-analyze-agent-a-4e/report.md`)*
3. *Jev / TypeSafe feasibility research (`data/research-jev-and-typesafe-model-feasibil-bd/report.md`)*