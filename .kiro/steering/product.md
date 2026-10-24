# Product: BridgeTriage

Built for Hackachusetts R6.

## The problem

Routine bridge inspections are fixed by federal rules, so their timing is not a
choice we get to make. What we *can* choose is where to spend a limited budget
for **N extra drone / in-depth inspections per year**. The question is simple:
of all the bridges in the state, which N should get that extra scrutiny this
year to buy the most safety for the money?

## What BridgeTriage does

It ranks every bridge by **risk reduction per dollar** and picks the top N.

For each bridge we compute:

```
score = p_worse * consequence / insp_cost
```

- **p_worse** — probability the bridge is *worse now than its last recorded
  rating*. Driven by age, material, current rating, months since last
  inspection, and decline rates the model learns from history.
- **consequence** — how much it matters if this bridge fails: traffic volume
  (ADT), and whether it crosses rail or water.
- **insp_cost** — estimated cost of the extra inspection. Dividing by it turns
  the ranking into "most risk reduction per dollar," not just "most risky."

We pick the N bridges with the highest score.

## How we prove it works

A ranking is only credible if it would have caught real problems. So we run a
backtest:

1. **Pick with 2018 data** using our planner and two baselines:
   - worst-rating-first
   - oldest-first
2. **Check against 2022** — which bridges actually got worse over those four
   years?
3. Compare how many real decliners each strategy caught. Our planner should
   beat both baselines. The result is a proof chart.

## Rules (non-negotiable)

- **Real data only.** FHWA NBI Massachusetts data, nothing else. Never invent
  bridges, ratings, or any field value. If data is missing, say so — do not
  fill it with a guess dressed up as real.
- **No lookahead.** Never fit the decline model on years *after* the pick year.
  Picking with 2018 means the model only sees data through 2018. Leaking future
  data invalidates the whole proof.
- **NBI "N" means not applicable, not zero.** A culvert bridge has "N" for deck
  because it has no deck — that is not a rating of 0. Treat "N" as missing /
  not-applicable, never as a numeric low score.
- **Every score carries a one-line reason.** No bridge gets ranked without a
  short human-readable explanation of why (e.g. "old timber, ADT 18k, 54 mo
  since last inspection").
- **Costs are estimates.** `insp_cost` is a modeled estimate, not a quoted
  price. Label it that way anywhere it surfaces to a user.
