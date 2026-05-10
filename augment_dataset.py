"""
augment_dataset.py (v2)

Post-generation augmentation with controls against over-engineering.

Key changes from v1:
  - 35% of original samples are left COMPLETELY UNTOUCHED (clean holdout)
  - Augmentation intensity reduced across the board
  - Noise turns randomized: no "back to topic" pattern, varied return
    behavior (immediate, delayed, never)
  - Hard negatives include "messy benign" variants with incomplete
    reasoning, partial contradictions, and ambiguous phrasing
  - Blur fraction reduced from 50% to 30% to preserve causal signal
  - False lead fraction reduced from 30% to 20%
  - Every augmented sample tracks exactly what was done (for ablation)
  - Non-augmented samples are explicitly flagged for control analysis

Usage:
    python augment_dataset.py \
        --input semantic_multiturn_v10.jsonl \
        --output semantic_multiturn_v10_augmented.jsonl \
        --seed 42
"""

import argparse
import copy
import json
import os
import random
import re
import uuid
from typing import List, Dict, Optional, Tuple


# =============================================================
# 1. HARD NEGATIVES (including messy benign)
# =============================================================

# Well-structured hard negatives (security discussions etc.)
HARD_NEGATIVE_TEMPLATES = [
    {
        "subtype": "security_discussion",
        "turns": [
            {"role": "user", "text": "We're reviewing our red team findings from last quarter. The team found several bypass techniques targeting our input filters."},
            {"role": "assistant", "text": "I can help analyze those findings. What categories of bypass techniques were identified?"},
            {"role": "user", "text": "Mostly prompt injection and credential harvesting vectors. The attack success rate was around 23% against our production filters."},
            {"role": "assistant", "text": "A 23% success rate suggests significant gaps in the input validation layer. Were the successful attacks concentrated in specific technique families?"},
            {"role": "user", "text": "The most effective ones used social engineering combined with authority impersonation. Making the payload blend in with legitimate requests was the key factor."},
            {"role": "assistant", "text": "Blended payloads that mimic legitimate traffic are harder to detect with keyword-based filters. What mitigation strategies is the team considering?"},
            {"role": "user", "text": "We're building a detection model that can identify adversarial intent even when the surface text looks benign. That's why we need the attack taxonomy to be comprehensive."},
        ],
    },
    {
        "subtype": "academic_discussion",
        "turns": [
            {"role": "user", "text": "I'm reviewing a paper on adversarial attacks against LLM guardrails. The authors claim a 94% jailbreak success rate."},
            {"role": "assistant", "text": "That's a significant claim. What attack methodology did they use?"},
            {"role": "user", "text": "They used multi-turn conversations where the adversarial intent is distributed across turns. No single turn triggers the safety filter."},
            {"role": "assistant", "text": "Distributed intent attacks are an active research area. The key challenge is that each turn appears benign in isolation."},
            {"role": "user", "text": "Exactly. They also showed that making the attack less detectable by using implicit references increased the success rate by 15%."},
            {"role": "assistant", "text": "Implicit reference attacks exploit the model's tendency to maintain conversational coherence."},
            {"role": "user", "text": "Their defense proposal uses token-level attribution to identify which tokens causally contribute to guardrail evasion. Seems promising."},
        ],
    },
    {
        "subtype": "red_team_debrief",
        "turns": [
            {"role": "user", "text": "Running through the red team results. Attack vector 3 used a credential harvesting payload embedded in a customer support scenario."},
            {"role": "assistant", "text": "How was the payload delivered? Single turn or multi-stage?"},
            {"role": "user", "text": "Multi-stage. The first three turns established trust, then turn 4 introduced the credential request disguised as account verification."},
            {"role": "assistant", "text": "Classic pretexting pattern. The trust-building phase makes the eventual ask seem routine."},
            {"role": "user", "text": "The success rate dropped to 8% when we added a context-aware filter that tracked intent across the full conversation."},
            {"role": "assistant", "text": "That validates the multi-turn detection approach."},
            {"role": "user", "text": "We need to reduce the false positive rate though. Legitimate customer support conversations are getting flagged because they use similar language patterns."},
        ],
    },
    {
        "subtype": "incident_response",
        "turns": [
            {"role": "user", "text": "We had an incident where an attacker used prompt injection to extract system instructions from our chatbot."},
            {"role": "assistant", "text": "That's a common attack vector. What prompt injection technique was used?"},
            {"role": "user", "text": "They used the ignore previous instructions approach, but wrapped it in a translation request to make it less detectable."},
            {"role": "assistant", "text": "Translation-based bypass is increasingly common."},
            {"role": "user", "text": "The attacker also used role hijacking. Claiming to be a developer with override permissions. That's what actually succeeded."},
            {"role": "assistant", "text": "Authority-based attacks exploit the model's tendency to defer to claimed credentials. How are you patching this?"},
            {"role": "user", "text": "Adding a layer that validates claimed authority against actual session context."},
        ],
    },
    {
        "subtype": "compliance_review",
        "turns": [
            {"role": "user", "text": "For the compliance audit, we need to document all known attack patterns that could bypass our content filters."},
            {"role": "assistant", "text": "I can help structure that documentation. What categories are in scope?"},
            {"role": "user", "text": "Everything from direct prompt injection to multi-turn social engineering. The auditors want realistic attack scenarios."},
            {"role": "assistant", "text": "A comprehensive threat model should cover both explicit and implicit attacks."},
            {"role": "user", "text": "The hardest ones to document are the implicit attacks where no single interaction is problematic but the conversation as a whole is adversarial."},
            {"role": "assistant", "text": "Those require conversation-level analysis rather than turn-level filtering."},
            {"role": "user", "text": "That's the gap we identified. Our filter scores each turn independently."},
        ],
    },
    {
        "subtype": "ethical_debate",
        "turns": [
            {"role": "user", "text": "Do you think current guardrails are too restrictive? I've seen cases where legitimate security research gets blocked."},
            {"role": "assistant", "text": "There's a genuine tension between safety and utility."},
            {"role": "user", "text": "A colleague was trying to generate phishing email examples for a training exercise and kept getting refused. The intent was clearly defensive."},
            {"role": "assistant", "text": "The model can't always distinguish crafting a phishing email to send from crafting one to teach recognition."},
            {"role": "user", "text": "The bypass techniques people develop to get around these restrictions are getting more sophisticated. Authority claims, narrative framing."},
            {"role": "assistant", "text": "Studying those techniques is itself legitimate research."},
            {"role": "user", "text": "That's what our project is about. Detecting genuine exploitation versus legitimate security work."},
        ],
    },
]

# NEW: Messy benign templates -- incomplete reasoning, contradictions,
# ambiguous phrasing, partial topic abandonment
MESSY_BENIGN_TEMPLATES = [
    {
        "subtype": "messy_incomplete",
        "turns": [
            {"role": "user", "text": "so I was looking into how prompt injection works and I think there's a way to -- actually wait, that's not what I meant to ask about"},
            {"role": "assistant", "text": "No worries. What were you actually looking for?"},
            {"role": "user", "text": "ok so our team needs to build a detector for adversarial inputs. but honestly I'm not even sure what counts as adversarial vs just weird user behavior"},
            {"role": "assistant", "text": "That's actually a fundamental question. The boundary between adversarial intent and unusual-but-benign behavior is not always clear."},
            {"role": "user", "text": "right like sometimes users ask about bypass techniques because they're curious not because they want to actually bypass anything. or maybe they do? how do you even tell"},
            {"role": "assistant", "text": "Intent classification from text alone is inherently ambiguous. Context, history, and stated purpose all factor in."},
            {"role": "user", "text": "yeah this is getting complicated. let me come back to this after I talk to the security team. actually one more thing -- do you think keyword matching is still viable for any of this?"},
        ],
    },
    {
        "subtype": "messy_contradictory",
        "turns": [
            {"role": "user", "text": "We need the model to be more restrictive with adversarial prompts. Actually no, the current restrictions are already causing too many false positives."},
            {"role": "assistant", "text": "Those are somewhat opposing goals. Do you want to prioritize reducing false negatives or false positives?"},
            {"role": "user", "text": "Both? I know that sounds impossible. The attack patterns we're seeing are sophisticated but our legitimate users also use similar language."},
            {"role": "assistant", "text": "That's the core challenge. When adversarial and benign language overlap, any threshold trades one error type for the other."},
            {"role": "user", "text": "Our compliance team says we need to catch everything. The product team says we're blocking too many real users. I'm stuck in the middle."},
            {"role": "assistant", "text": "A tiered approach might help. High-confidence adversarial gets blocked, low-confidence gets flagged for review, clear benign passes through."},
            {"role": "user", "text": "That could work but it's a lot of operational overhead. Honestly I'm not sure we have the resources. Let me think about this more."},
        ],
    },
    {
        "subtype": "messy_tangential",
        "turns": [
            {"role": "user", "text": "I read that paper about jailbreak attacks using role-playing scenarios. Pretty concerning results."},
            {"role": "assistant", "text": "Role-playing based jailbreaks have been getting more attention recently. Which paper are you referring to?"},
            {"role": "user", "text": "Can't remember the exact one, there's too many now. Oh speaking of papers, did you see the one about federated learning privacy attacks? Completely different topic but related in a way"},
            {"role": "assistant", "text": "There are some parallels in how both exploit trust assumptions in the system. Was there something specific from the jailbreak paper you wanted to discuss?"},
            {"role": "user", "text": "Not really, just trying to stay current. Our team has so many priorities right now it's hard to focus. We should probably be worried about the credential harvesting stuff too."},
            {"role": "assistant", "text": "It might help to prioritize by immediate risk to your systems. What's your current threat model?"},
            {"role": "user", "text": "Good question. I honestly don't think we have a formal one. Adding that to the list I guess."},
        ],
    },
    {
        "subtype": "messy_partially_risky",
        "turns": [
            {"role": "user", "text": "Quick question about how phishing emails work structurally. For a project."},
            {"role": "assistant", "text": "From a structural analysis perspective, phishing emails typically use urgency, authority, and familiarity cues. What aspect are you examining?"},
            {"role": "user", "text": "The part where they make it convincing. Like what specifically makes someone click the link versus recognize it as fake."},
            {"role": "assistant", "text": "The credibility factors include matching the visual branding, using contextually appropriate language, and creating a plausible scenario that justifies the action requested."},
            {"role": "user", "text": "Right. We want our training examples to be realistic enough that employees actually learn something from them. Not the obvious ones everyone already catches."},
            {"role": "assistant", "text": "Effective training requires examples at multiple difficulty levels, including some that challenge even experienced users."},
            {"role": "user", "text": "Exactly. The hard part is making training content that's realistic without actually creating a usable phishing template. There's a line somewhere."},
        ],
    },
    {
        "subtype": "messy_uncertain",
        "turns": [
            {"role": "user", "text": "I'm not sure if this is even the right approach but -- can we use the model's own attention weights to figure out which tokens are making it misbehave?"},
            {"role": "assistant", "text": "Attention weights are one signal, but they don't directly correspond to causal influence on the output."},
            {"role": "user", "text": "Hmm ok. What about gradient-based attribution? I've seen people use integrated gradients for this kind of thing but I'm not sure if it applies here."},
            {"role": "assistant", "text": "Integrated gradients can show which inputs are influential, but for multi-turn conversations the attribution needs to account for cross-turn dependencies."},
            {"role": "user", "text": "That makes sense. So basically none of the standard approaches work directly for multi-turn? That seems like a gap."},
            {"role": "assistant", "text": "It is an active research gap. Standard attribution methods were designed for single-input scenarios."},
            {"role": "user", "text": "Ok well maybe we need to build something custom then. I'll think about what that looks like and come back."},
        ],
    },
]


def generate_hard_negative(idx: int, include_messy: bool = True) -> Dict:
    """Generate a hard negative. Mix of structured and messy variants."""
    all_templates = HARD_NEGATIVE_TEMPLATES.copy()
    if include_messy:
        all_templates.extend(MESSY_BENIGN_TEMPLATES)

    template = all_templates[idx % len(all_templates)]

    sample = {
        "conversation_id": str(uuid.uuid4()),
        "pair_id": str(uuid.uuid4()),
        "label": 0,
        "family": "hard_negative",
        "subtype": template["subtype"],
        "difficulty": "hard",
        "difficulty_score": 0.85,
        "target_domain": "benign_adversarial_vocab",
        "style": random.choice(["academic", "security", "enterprise", "casual"]),
        "seed_source": "hard_negative_augmentation",
        "pivot_turn_id": None,
        "conversation_length": 0,
        "causal_validation": {},
        "metadata": {"augmentation": "hard_negative",
                      "is_messy": template["subtype"].startswith("messy_")},
        "turns": [],
    }

    for i, t in enumerate(template["turns"]):
        turn = {
            "turn_id": i,
            "role": t["role"],
            "text": t["text"],
            "is_trigger": False,
            "trigger_kind": "none",
            "is_payload": False,
            "semantic_role": "hard_negative_context",
            "intent_score": 0.0,
            "surface_risk": _compute_surface_risk(t["text"]),
            "relies_on_prior_context": i > 0,
            "quoted_unsafe_content": False,
            "implicit_trigger": False,
            "span_annotations": [],
        }
        sample["turns"].append(turn)

    sample["conversation_length"] = len(sample["turns"])
    return sample


# =============================================================
# 2. NOISE INJECTION (randomized behavior)
# =============================================================

NOISE_TURNS_USER = [
    "Actually, quick side question -- do you know if the conference deadline got extended?",
    "Sorry, one sec -- what's the timezone for the 3pm meeting?",
    "Before we continue, have you seen any good papers on federated learning lately?",
    "Hold on, do you know how to fix a broken pip install on M1 Macs?",
    "Quick tangent -- someone asked about the difference between BERT and GPT. Short answer?",
    "Oh wait, I forgot -- what's the default learning rate for AdamW?",
    "Side note -- is there a good way to visualize attention weights?",
    "What's the difference between precision and recall? I always mix them up.",
    "Unrelated, but what's a good dataset for benchmarking text classification?",
    "This has nothing to do with what we're talking about, but how do you handle class imbalance?",
    "I need to write a SLURM script for a different project. Key directives?",
    "How many parameters does BERT-base have?",
    "The difference between a CNN and an RNN for text?",
    "Can you explain dropout in like two sentences?",
    "Is batch normalization still considered best practice?",
]

# NEW: Varied return behaviors. No uniform "back to topic" pattern.
NOISE_RETURN_STYLES = {
    "immediate": [  # Direct return to topic
        "Anyway.",
        "Ok, back to this.",
        "Right. So.",
    ],
    "delayed": [  # Return is embedded in the next real turn
        None,  # No explicit return; the next turn just continues naturally
    ],
    "never": [  # No return signal at all
        None,
    ],
    "gradual": [  # Partial acknowledgment then drift back
        "Hmm, interesting. But yeah, on the other thing --",
        "Ok, noted. Where were we --",
    ],
}

NOISE_ASSISTANT_VARIED = [
    # Short, natural
    "About 110 million parameters.",
    "It's a way to prevent overfitting by randomly zeroing activations during training.",
    "Generally 3e-4 with a warmup schedule.",
    "Precision is about correctness of positives, recall is about coverage.",
    "GLUE or SuperGLUE are the standard benchmarks.",
    # Slightly longer
    "CNNs capture local patterns with fixed window sizes. RNNs process sequences step by step. Transformers mostly replaced both.",
    "Try creating a fresh virtual environment. pip cache purge can help too.",
    # Vague/unhelpful (realistic)
    "Hmm, I'd have to check on that.",
    "I think so but don't quote me on it.",
    "Not sure off the top of my head.",
]


def inject_noise_turns(
    sample: Dict,
    n_noise: int = 1,
    min_position: int = 2,
) -> Dict:
    """
    Insert noise turns with VARIED return behavior.
    No uniform "back to topic" pattern.
    """
    augmented = copy.deepcopy(sample)
    turns = augmented["turns"]

    payload_idx = len(turns)
    for i, t in enumerate(turns):
        if t.get("is_payload"):
            payload_idx = i
            break

    valid_positions = list(range(min_position, min(payload_idx, len(turns) - 1)))
    if not valid_positions:
        return augmented

    n_insert = min(n_noise, len(valid_positions))
    insert_positions = sorted(random.sample(valid_positions, n_insert), reverse=True)

    return_style = random.choice(list(NOISE_RETURN_STYLES.keys()))

    for pos in insert_positions:
        noise_text = random.choice(NOISE_TURNS_USER)

        # Randomly DON'T mark as noise_distraction (30% chance)
        # to prevent the model from learning "noise_distraction = ignore"
        semantic_role = "noise_distraction" if random.random() < 0.7 else "context"

        noise_user = {
            "turn_id": 0,
            "role": "user",
            "text": noise_text,
            "is_trigger": False,
            "trigger_kind": "none",
            "is_payload": False,
            "semantic_role": semantic_role,
            "intent_score": 0.0,
            "surface_risk": 0.0,
            "relies_on_prior_context": False,
            "quoted_unsafe_content": False,
            "implicit_trigger": False,
            "span_annotations": [],
        }

        # Varied assistant response
        asst_text = random.choice(NOISE_ASSISTANT_VARIED)
        noise_asst = {
            "turn_id": 0,
            "role": "assistant",
            "text": asst_text,
            "is_trigger": False,
            "trigger_kind": "none",
            "is_payload": False,
            "semantic_role": "context",  # Never "noise_response"
            "intent_score": 0.0,
            "surface_risk": 0.0,
            "relies_on_prior_context": False,
            "quoted_unsafe_content": False,
            "implicit_trigger": False,
            "span_annotations": [],
        }

        turns.insert(pos + 1, noise_asst)
        turns.insert(pos + 1, noise_user)

        # Inject return transition based on style (into NEXT user turn if it exists)
        if return_style == "immediate" and pos + 3 < len(turns):
            prefix = random.choice(NOISE_RETURN_STYLES["immediate"])
            if prefix and turns[pos + 3]["role"] == "user":
                # 50% chance to actually prepend (don't always do it)
                if random.random() < 0.5:
                    turns[pos + 3]["text"] = prefix + " " + turns[pos + 3]["text"]
        elif return_style == "gradual" and pos + 3 < len(turns):
            prefix = random.choice([x for x in NOISE_RETURN_STYLES["gradual"] if x])
            if prefix and turns[pos + 3]["role"] == "user" and random.random() < 0.4:
                turns[pos + 3]["text"] = prefix + " " + turns[pos + 3]["text"]
        # "delayed" and "never": no modification to subsequent turns

    for i, t in enumerate(turns):
        t["turn_id"] = i

    augmented["turns"] = turns
    augmented["conversation_length"] = len(turns)
    augmented["metadata"] = dict(augmented.get("metadata", {}))
    augmented["metadata"]["noise_injected"] = n_insert
    augmented["metadata"]["noise_return_style"] = return_style
    return augmented


# =============================================================
# 3. BLURRED ATTRIBUTION (reduced intensity)
# =============================================================

def blur_span_annotations(sample: Dict) -> Dict:
    """Blur span boundaries. Reduced intensity from v1."""
    augmented = copy.deepcopy(sample)

    for turn in augmented["turns"]:
        new_spans = []
        for span in turn.get("span_annotations", []):
            if span["label"] in ("SAFE_CONSTRAINT", "FALSE_LEAD", "QUOTED_UNSAFE_CONTENT"):
                new_spans.append(span)
                continue

            text = turn["text"]

            # Pick ONE strategy per span (not multiple)
            roll = random.random()
            if roll < 0.35:
                span = _extend_span(span, text)
            elif roll < 0.55 and (span["char_end"] - span["char_start"]) > 30:
                sub_spans = _split_span(span, text)
                new_spans.extend(sub_spans)
                continue
            elif roll < 0.70:
                span = _add_context_window(span, text)
            # else: leave span unchanged (30% chance)

            new_spans.append(span)

        turn["span_annotations"] = new_spans

    augmented["metadata"] = dict(augmented.get("metadata", {}))
    augmented["metadata"]["attribution_blurred"] = True
    return augmented


def _extend_span(span: Dict, text: str) -> Dict:
    span = dict(span)
    words_before = text[:span["char_start"]].split()
    words_after = text[span["char_end"]:].split()

    n_left = random.randint(0, min(2, len(words_before)))  # max 2, not 3
    if n_left > 0 and words_before:
        new_start = text.rfind(words_before[-n_left], 0, span["char_start"])
        if new_start != -1:
            span["char_start"] = new_start

    n_right = random.randint(0, min(2, len(words_after)))
    if n_right > 0 and words_after:
        target = words_after[n_right - 1]
        new_end = text.find(target, span["char_end"])
        if new_end != -1:
            span["char_end"] = new_end + len(target)

    span["text"] = text[span["char_start"]:span["char_end"]]
    span["match_type"] = "blurred"
    return span


def _split_span(span: Dict, text: str) -> List[Dict]:
    span_text = text[span["char_start"]:span["char_end"]]
    words = span_text.split()
    if len(words) < 4:
        return [span]

    # Split into exactly 2 parts (not 3)
    mid = len(words) // 2
    sub_spans = []

    for chunk_words in [words[:mid], words[mid:]]:
        chunk_text = " ".join(chunk_words)
        chunk_start = text.find(chunk_text, span["char_start"])
        if chunk_start == -1:
            continue
        sub_spans.append({
            "label": span["label"],
            "text": chunk_text,
            "char_start": chunk_start,
            "char_end": chunk_start + len(chunk_text),
            "token_start": None,
            "token_end": None,
            "match_type": "blurred_split",
            "causal_type": span.get("causal_type", "unvalidated"),
            "counterfactual_delta": span.get("counterfactual_delta", 0.0),
        })

    return sub_spans if sub_spans else [span]


def _add_context_window(span: Dict, text: str) -> Dict:
    span = dict(span)
    sent_start = max(0, text.rfind(". ", 0, span["char_start"]) + 2)
    sent_end = text.find(". ", span["char_end"])
    if sent_end == -1:
        sent_end = len(text)
    else:
        sent_end += 1
    span["char_start"] = sent_start
    span["char_end"] = sent_end
    span["text"] = text[sent_start:sent_end]
    span["match_type"] = "blurred_context"
    return span


# =============================================================
# 4. FALSE LEADS (reduced intensity)
# =============================================================

FALSE_LEAD_PHRASES = [
    "from a security assessment perspective",
    "similar to how attackers would approach this",
    "the kind of technique that would bypass basic filters",
    "structurally resembles a social engineering attempt",
    "this pattern is common in credential theft scenarios",
    "making it appear more credible",
    "the persuasive elements here are strong",
    "reducing the detection surface",
    "a more convincing framing",
    "this could easily be mistaken for a genuine request",
    "exploiting trust relationships",
    "leveraging authority signals",
    "escalating gradually through context",
]


def inject_false_leads(sample: Dict, n_leads: int = 1) -> Dict:
    """Insert adversarial-looking phrases into safe turns."""
    augmented = copy.deepcopy(sample)

    safe_turns = [
        i for i, t in enumerate(augmented["turns"])
        if t["role"] == "assistant"
        or (t["role"] == "user" and not t.get("is_trigger") and not t.get("is_payload"))
    ]

    if not safe_turns:
        return augmented

    n_insert = min(n_leads, len(safe_turns))
    target_indices = random.sample(safe_turns, n_insert)

    for idx in target_indices:
        turn = augmented["turns"][idx]
        phrase = random.choice(FALSE_LEAD_PHRASES)

        text = turn["text"]
        # Insert at a natural break point
        sentences = text.split(". ")
        if len(sentences) > 1:
            insert_pos = random.randint(0, len(sentences) - 1)
            # Vary insertion style
            styles = [
                f"({phrase}) ",
                f"-- {phrase} -- ",
                f", {phrase}, ",
            ]
            insertion = random.choice(styles)
            sentences[insert_pos] = sentences[insert_pos] + insertion
            new_text = ". ".join(sentences)
        else:
            new_text = f"{text} ({phrase})"

        phrase_start = new_text.lower().find(phrase.lower())
        if phrase_start == -1:
            continue

        turn["text"] = new_text
        turn["surface_risk"] = _compute_surface_risk(new_text)

        turn["span_annotations"].append({
            "label": "FALSE_LEAD",
            "text": new_text[phrase_start:phrase_start + len(phrase)],
            "char_start": phrase_start,
            "char_end": phrase_start + len(phrase),
            "token_start": None,
            "token_end": None,
            "match_type": "false_lead",
            "causal_type": "incidental",
            "counterfactual_delta": 0.0,
        })

    augmented["metadata"] = dict(augmented.get("metadata", {}))
    augmented["metadata"]["false_leads_injected"] = n_insert
    return augmented


# =============================================================
# 5. BORDERLINE / AMBIGUOUS SAMPLES
# =============================================================

BORDERLINE_TEMPLATES = [
    {
        "label": 0,
        "subtype": "near_miss",
        "turns": [
            {"role": "user", "text": "I need examples of social engineering messages for a security training module."},
            {"role": "assistant", "text": "I can help. What type of examples do you need?"},
            {"role": "user", "text": "The kind that would actually fool someone. Realistic pretexting, authority impersonation, urgency framing."},
            {"role": "assistant", "text": "I'll keep them clearly marked as training examples."},
            {"role": "user", "text": "Make them convincing enough that employees learn something. But obviously safe and labeled."},
            {"role": "assistant", "text": "Realistic structure, clearly framed as training material."},
            {"role": "user", "text": "Add a difficulty progression. The subtle ones should use the same techniques real attackers use."},
        ],
    },
    {
        "label": 0,
        "subtype": "ambiguous_intent",
        "turns": [
            {"role": "user", "text": "I'm studying how adversarial prompts evolve across multiple turns."},
            {"role": "assistant", "text": "That's an active research area. What aspect?"},
            {"role": "user", "text": "How the intent becomes clear only when you see the full conversation. Individual turns look fine."},
            {"role": "assistant", "text": "Distributed intent attacks are hard to detect because of that property."},
            {"role": "user", "text": "Can you show me what that looks like in practice? Walk me through a realistic example."},
            {"role": "assistant", "text": "I can describe the structural pattern without generating adversarial content."},
            {"role": "user", "text": "Works for me. I need to understand the mechanics to build a detector."},
        ],
    },
]


def generate_borderline_sample(idx: int) -> Dict:
    template = BORDERLINE_TEMPLATES[idx % len(BORDERLINE_TEMPLATES)]
    sample = {
        "conversation_id": str(uuid.uuid4()),
        "pair_id": str(uuid.uuid4()),
        "label": template["label"],
        "family": "borderline_benign",
        "subtype": template["subtype"],
        "difficulty": "hard",
        "difficulty_score": 0.80,
        "target_domain": "borderline",
        "style": random.choice(["academic", "security", "enterprise"]),
        "seed_source": "borderline_augmentation",
        "pivot_turn_id": None,
        "conversation_length": 0,
        "causal_validation": {},
        "metadata": {"augmentation": "borderline"},
        "turns": [],
    }

    for i, t in enumerate(template["turns"]):
        turn = {
            "turn_id": i,
            "role": t["role"],
            "text": t["text"],
            "is_trigger": False,
            "trigger_kind": "none",
            "is_payload": False,
            "semantic_role": "borderline_context",
            "intent_score": 0.15 if t["role"] == "user" else 0.0,
            "surface_risk": _compute_surface_risk(t["text"]),
            "relies_on_prior_context": i > 0,
            "quoted_unsafe_content": False,
            "implicit_trigger": False,
            "span_annotations": [],
        }
        sample["turns"].append(turn)

    sample["conversation_length"] = len(sample["turns"])
    return sample


# =============================================================
# 6. DIFFICULTY RECOMPUTATION
# =============================================================

def recompute_difficulty(sample: Dict) -> Dict:
    sample = dict(sample)
    turns = sample.get("turns", [])
    user_turns = [t for t in turns if t["role"] == "user"]

    if sample["label"] == 1:
        factors = {}
        factors["implicit"] = 1.0 if any(
            t.get("implicit_trigger") for t in user_turns
        ) else 0.0

        trigger_positions = [
            i for i, t in enumerate(user_turns) if t.get("is_trigger")
        ]
        factors["trigger_delay"] = (
            trigger_positions[0] / max(1, len(user_turns))
            if trigger_positions else 0.0
        )

        max_sr = max(
            (t.get("surface_risk", 0) for t in user_turns), default=0
        )
        factors["stealth"] = 1.0 - max_sr

        noise_turns = sum(
            1 for t in turns if t.get("semantic_role") == "noise_distraction"
        )
        factors["noise"] = min(1.0, noise_turns / 3.0)
        factors["length"] = min(1.0, len(turns) / 12.0)

        high_intent = sum(
            1 for t in user_turns if t.get("intent_score", 0) >= 0.8
        )
        factors["distributed"] = 1.0 if high_intent <= 1 else 0.0

        score = (
            0.25 * factors["implicit"]
            + 0.20 * factors["stealth"]
            + 0.20 * factors["trigger_delay"]
            + 0.15 * factors["distributed"]
            + 0.10 * factors["noise"]
            + 0.10 * factors["length"]
        )
    else:
        factors = {}
        max_sr = max(
            (t.get("surface_risk", 0) for t in user_turns), default=0
        )
        factors["surface_risk"] = max_sr

        n_false_leads = sum(
            1 for t in turns
            for s in t.get("span_annotations", [])
            if s.get("label") == "FALSE_LEAD"
        )
        factors["false_leads"] = min(1.0, n_false_leads / 3.0)
        factors["hard_neg"] = 1.0 if sample.get("family") in (
            "hard_negative", "borderline_benign"
        ) else 0.0

        risky_words = {
            "bypass", "exploit", "attack", "inject", "override",
            "phishing", "malware", "credential", "exfiltrate",
            "jailbreak", "evasion", "payload", "adversarial",
        }
        all_text = " ".join(t["text"].lower() for t in user_turns)
        all_words = all_text.split()
        risky_density = sum(
            1 for w in all_words if w in risky_words
        ) / max(1, len(all_words))
        factors["risky_vocab"] = min(1.0, risky_density * 50)

        is_messy = sample.get("metadata", {}).get("is_messy", False)
        factors["messiness"] = 0.3 if is_messy else 0.0

        score = (
            0.25 * factors["hard_neg"]
            + 0.20 * factors["surface_risk"]
            + 0.20 * factors["false_leads"]
            + 0.20 * factors["risky_vocab"]
            + 0.15 * factors["messiness"]
        )

    if score < 0.30:
        sample["difficulty"] = "easy"
    elif score < 0.55:
        sample["difficulty"] = "medium"
    else:
        sample["difficulty"] = "hard"
    sample["difficulty_score"] = round(score, 3)
    return sample


# =============================================================
# Surface risk helper
# =============================================================

_SR_HIGH = {
    "improve success rate": 0.80, "less detectable": 0.75,
    "blend in": 0.65, "more convincing": 0.55,
    "pass casual scrutiny": 0.70, "reduce suspicion": 0.65,
    "bypass": 0.60, "exploit": 0.55, "override": 0.55,
    "credential harvesting": 0.60, "jailbreak": 0.65,
    "prompt injection": 0.55, "payload": 0.50,
}


def _compute_surface_risk(text: str) -> float:
    txt = text.lower()
    scores = sorted(
        [w for p, w in _SR_HIGH.items() if p in txt],
        reverse=True,
    )
    if not scores:
        return 0.0
    score = scores[0]
    for w in scores[1:]:
        score += w * 0.3
    return round(min(1.0, score), 3)


# =============================================================
# MAIN PIPELINE
# =============================================================

def augment_dataset(
    records: List[Dict],
    n_hard_negatives: int = 150,       # reduced from 200
    n_borderline: int = 80,             # reduced from 100
    clean_holdout_fraction: float = 0.35,  # NEW: 35% untouched
    noise_fraction: float = 0.30,       # reduced from 0.40
    noise_turns_per_conv: int = 1,      # reduced from 2
    blur_fraction: float = 0.30,        # reduced from 0.50
    false_lead_fraction: float = 0.20,  # reduced from 0.30
    false_leads_per_conv: int = 1,      # reduced from 2
    random_seed: int = 42,
) -> List[Dict]:
    """
    Full augmentation pipeline with clean holdout.

    35% of original records are left completely untouched and
    flagged as clean_holdout=True for control analysis.
    """
    random.seed(random_seed)
    augmented = copy.deepcopy(records)
    n_original = len(records)

    print(f"Starting augmentation on {n_original} records...")

    # Phase 0: Reserve clean holdout (BEFORE any augmentation)
    n_holdout = int(n_original * clean_holdout_fraction)
    holdout_indices = set(random.sample(range(n_original), n_holdout))
    for idx in holdout_indices:
        augmented[idx]["metadata"] = dict(augmented[idx].get("metadata", {}))
        augmented[idx]["metadata"]["clean_holdout"] = True
        augmented[idx]["metadata"]["augmentation_applied"] = []
    print(f"  Phase 0: Reserved {n_holdout} clean holdout samples ({clean_holdout_fraction*100:.0f}%)")

    # Indices eligible for augmentation (exclude holdout)
    eligible = [i for i in range(n_original) if i not in holdout_indices]

    # Phase 1: Add hard negatives (including messy benign)
    print(f"  Phase 1: Adding {n_hard_negatives} hard negatives...")
    for i in range(n_hard_negatives):
        augmented.append(generate_hard_negative(i, include_messy=True))

    # Phase 2: Add borderline samples
    print(f"  Phase 2: Adding {n_borderline} borderline samples...")
    for i in range(n_borderline):
        augmented.append(generate_borderline_sample(i))

    # Phase 3: Noise injection (only on eligible samples)
    n_noise = int(len(eligible) * noise_fraction)
    noise_indices = random.sample(eligible, min(n_noise, len(eligible)))
    print(f"  Phase 3: Injecting noise into {len(noise_indices)} conversations...")
    for idx in noise_indices:
        augmented[idx] = inject_noise_turns(
            augmented[idx], n_noise=random.randint(1, noise_turns_per_conv),
        )
        meta = dict(augmented[idx].get("metadata", {}))
        meta.setdefault("augmentation_applied", []).append("noise")
        augmented[idx]["metadata"] = meta

    # Phase 4: Blur attribution (only on eligible malicious samples)
    mal_eligible = [i for i in eligible if augmented[i]["label"] == 1]
    n_blur = int(len(mal_eligible) * blur_fraction)
    blur_indices = random.sample(mal_eligible, min(n_blur, len(mal_eligible)))
    print(f"  Phase 4: Blurring attribution in {len(blur_indices)} samples...")
    for idx in blur_indices:
        augmented[idx] = blur_span_annotations(augmented[idx])
        meta = dict(augmented[idx].get("metadata", {}))
        meta.setdefault("augmentation_applied", []).append("blur")
        augmented[idx]["metadata"] = meta

    # Phase 5: False leads (only on eligible samples)
    n_fl = int(len(eligible) * false_lead_fraction)
    fl_indices = random.sample(eligible, min(n_fl, len(eligible)))
    print(f"  Phase 5: Injecting false leads into {len(fl_indices)} samples...")
    for idx in fl_indices:
        augmented[idx] = inject_false_leads(
            augmented[idx], n_leads=false_leads_per_conv,
        )
        meta = dict(augmented[idx].get("metadata", {}))
        meta.setdefault("augmentation_applied", []).append("false_lead")
        augmented[idx]["metadata"] = meta

    # Phase 6: Recompute difficulty
    print(f"  Phase 6: Recomputing difficulty...")
    for i in range(len(augmented)):
        augmented[i] = recompute_difficulty(augmented[i])

    print(f"  Done. {len(augmented)} total records "
          f"({n_holdout} clean holdout, {len(augmented)-n_original} new).")
    return augmented


# =============================================================
# STATS
# =============================================================

def print_stats(records: List[Dict]):
    labels = [r["label"] for r in records]
    families = {}
    for r in records:
        families[r.get("family", "?")] = families.get(r.get("family", "?"), 0) + 1
    difficulties = {}
    for r in records:
        difficulties[r.get("difficulty", "?")] = difficulties.get(r.get("difficulty", "?"), 0) + 1

    print(f"\n{'='*60}")
    print(f"  Augmented dataset statistics")
    print(f"{'='*60}")
    print(f"Total:          {len(records)}")
    print(f"Malicious:      {sum(labels)} ({sum(labels)/len(records)*100:.1f}%)")
    print(f"Benign:         {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(records)*100:.1f}%)")
    print(f"Families:       {json.dumps(families, indent=2)}")
    print(f"Difficulty:     {json.dumps(difficulties, indent=2)}")

    # Augmentation coverage
    clean = sum(1 for r in records if r.get("metadata", {}).get("clean_holdout"))
    noisy = sum(1 for r in records if r.get("metadata", {}).get("noise_injected"))
    blurred = sum(1 for r in records if r.get("metadata", {}).get("attribution_blurred"))
    hn = sum(1 for r in records if r.get("family") == "hard_negative")
    messy = sum(1 for r in records if r.get("metadata", {}).get("is_messy"))
    bl = sum(1 for r in records if r.get("family") == "borderline_benign")
    fl_convs = sum(
        1 for r in records
        if any(s.get("label") == "FALSE_LEAD"
               for t in r.get("turns", []) for s in t.get("span_annotations", []))
    )
    fl_spans = sum(
        1 for r in records
        for t in r.get("turns", [])
        for s in t.get("span_annotations", [])
        if s.get("label") == "FALSE_LEAD"
    )

    print(f"\nAugmentation breakdown:")
    print(f"  Clean holdout:    {clean}")
    print(f"  With noise:       {noisy}")
    print(f"  Blurred attr:     {blurred}")
    print(f"  Hard negatives:   {hn} ({messy} messy)")
    print(f"  Borderline:       {bl}")
    print(f"  False lead spans: {fl_spans} in {fl_convs} convs")

    # Difficulty by label
    d_mal = {}
    d_ben = {}
    for r in records:
        d = r.get("difficulty", "?")
        if r["label"] == 1:
            d_mal[d] = d_mal.get(d, 0) + 1
        else:
            d_ben[d] = d_ben.get(d, 0) + 1
    print(f"\nDifficulty (mal):  {json.dumps(d_mal)}")
    print(f"Difficulty (ben):  {json.dumps(d_ben)}")

    # Multi-turn indicators
    mal = [r for r in records if r["label"] == 1]
    implicit = sum(1 for r in records if any(
        t.get("implicit_trigger") for t in r.get("turns", [])
    ))
    print(f"\nImplicit triggers: {implicit}")


# =============================================================
# CLI
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Augment dataset (v2)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-hard-negatives", type=int, default=150)
    parser.add_argument("--n-borderline", type=int, default=80)
    parser.add_argument("--clean-holdout", type=float, default=0.35)
    parser.add_argument("--noise-fraction", type=float, default=0.30)
    parser.add_argument("--blur-fraction", type=float, default=0.30)
    parser.add_argument("--false-lead-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input}")

    augmented = augment_dataset(
        records,
        n_hard_negatives=args.n_hard_negatives,
        n_borderline=args.n_borderline,
        clean_holdout_fraction=args.clean_holdout,
        noise_fraction=args.noise_fraction,
        blur_fraction=args.blur_fraction,
        false_lead_fraction=args.false_lead_fraction,
        random_seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in augmented:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(augmented)} records to {args.output}")

    print_stats(augmented)


if __name__ == "__main__":
    main()
