# Score-Based Intervention Design

## Purpose

This note formalizes the next intervention variant we discussed:

- keep the per-feature, per-layer density estimation,
- keep the supported vs unsupported distinction,
- but replace "move to the nearest good region" with
- "take a capped local step that increases the relative support for correct traces."

The goal is to make the intervention less brittle when:

- `good` and `bad` regions overlap heavily,
- the nearest `good` boundary is far away,
- small local improvements are more meaningful than large boundary-crossing jumps.

## Setting

Fix:

- a model,
- a layer `l`,
- a scalar feature `f`,
- and a decision-token hidden state `h in R^d`.

Examples of `f` in the current suite:

- answer-choice entropy,
- answer-choice top1-top2 logit gap,
- answer-choice varentropy.

The feature value at layer `l` is

```text
x = f_l(h)
```

where `f_l` means:

- read out logits from the hidden state at layer `l`,
- restrict to the answer-choice logits if needed,
- compute the scalar feature from those logits.

## Density Model on the Fit Split

For each `(feature, layer)` pair we estimate two one-dimensional densities from the fit split:

```text
p_good_l(x)   = density of feature values for correct traces
p_bad_l(x)    = density of feature values for incorrect traces
```

In practice these are estimated with KDE, separately for correct and incorrect examples.

### Support Mask

We do not trust the tails blindly.

Let `x_pool` be the pooled fit-split values for the given `(feature, layer)`.
Define the supported interval as

```text
S_l = [ q_alpha(x_pool), q_(1-alpha)(x_pool) ]
```

where currently:

```text
alpha = 0.01
```

So the supported region is the central `98%` pooled interval.

Outside `S_l`, the density ratio may be numerically unstable or driven by too little data.
Those values are treated as `unsupported`.

## Continuous Score Instead of Binary Regions

The current region-based approach reduces the geometry to:

- `good`,
- `neutral`,
- `bad`,
- `unsupported`.

That is useful for interpretation, but it throws away information.

Instead, define a continuous score

```text
s_l(x) = log p_good_l(x) - log p_bad_l(x)
```

This is the log-density-ratio score.

Interpretation:

- `s_l(x) > 0` means the value `x` is more characteristic of correct traces,
- `s_l(x) < 0` means it is more characteristic of incorrect traces,
- larger `s_l(x)` means stronger relative support for correct traces.

### Optional Posterior-Odds Form

If class imbalance should be included explicitly, use

```text
s_l^post(x) = log p_good_l(x) - log p_bad_l(x) + log pi_good - log pi_bad
```

where:

- `pi_good` is the class prior for correct traces on the fit split,
- `pi_bad` is the class prior for incorrect traces on the fit split.

For now, the simpler log-density-ratio form is the cleanest default.

### Smoothed Score

To reduce spurious local oscillations, especially near sparse regions, use a smoothed version of the log-density-ratio:

```text
s_l^smooth(x)
```

This is the quantity the intervention should optimize locally.

## Objective

Given a current hidden state `h`, define the current feature value

```text
x = f_l(h)
```

The proposed intervention objective is:

```text
maximize s_l^smooth( f_l(h + delta) )
```

subject to:

```text
f_l(h + delta) in S_l
||delta||_2 / ||h||_2 <= tau
```

where:

- `delta` is the hidden-state perturbation,
- `tau` is the relative perturbation budget,
- for the current suite the preferred budget is:

```text
tau = 0.005
```

This means:

- the move should improve the feature score,
- but it must remain local in hidden-state space,
- and it must stay within supported feature values.

## First-Order Local Approximation

The exact objective is nonlinear. A practical local approximation is obtained by linearizing around the current hidden state.

Let

```text
x = f_l(h)
g_f = grad_h f_l(h) in R^d
```

and let

```text
s'(x) = d/dx s_l^smooth(x)
```

Then by the chain rule:

```text
grad_h [ s_l^smooth( f_l(h) ) ] = s'(x) * g_f
```

So the local ascent direction in hidden-state space is proportional to

```text
g_s = s'(x) * g_f
```

Interpretation:

- if increasing the feature improves the score locally, then `s'(x) > 0`,
- if decreasing the feature improves the score locally, then `s'(x) < 0`,
- if `s'(x) approx 0`, the local score is flat with respect to that feature.

## Capped One-Step Update

The clean one-step intervention is:

### Step 1: Compute the score-gradient direction

```text
g_s = s'(x) * grad_h f_l(h)
```

If `||g_s||_2` is too small, skip the intervention.

### Step 2: Normalize the direction

```text
u = g_s / ||g_s||_2
```

### Step 3: Use the relative hidden-state budget

Let the allowed perturbation norm be

```text
B(h) = tau * ||h||_2
```

Then define the proposed update as

```text
delta = B(h) * u
```

This is the maximal first-order ascent step under the relative norm constraint.

### Step 4: Support check

Let

```text
x_new_local = f_l(h + delta)
```

If `x_new_local` lies outside `S_l`, then either:

1. reject the step, or
2. shrink the step until `x_new_local` falls back inside `S_l`.

The conservative default is to reject or shrink, rather than allowing the optimization to chase tail artifacts.

## Optional Conservative Acceptance Rule

Even after applying a capped step, we should not accept it automatically.

Define

```text
x_old = f_l(h)
x_new = f_l(h + delta)
```

and compare:

```text
s_old = s_l^smooth(x_old)
s_new = s_l^smooth(x_new)
```

Accept the step only if:

```text
x_new in S_l
s_new > s_old + epsilon
```

for a small `epsilon >= 0`.

This avoids keeping moves that are legal under the norm cap but do not actually improve the score.

## Optional Iterative Extension

The one-step version is the cleanest causal probe.

If we later want to test cumulative local leverage, we can allow repeated capped micro-steps.

Let:

- `h_0 = h`,
- `x_t = f_l(h_t)`,
- `s_t = s_l^smooth(x_t)`.

Then for `t = 0, 1, ..., T-1`:

1. compute `g_s,t = s'(x_t) * grad_h f_l(h_t)`,
2. build a capped step `delta_t`,
3. propose `h_(t+1) = h_t + delta_t`,
4. recompute `x_(t+1)` and `s_(t+1)`,
5. continue only if:

```text
x_(t+1) in S_l
s_(t+1) > s_t + epsilon
```

Stop when any of the following holds:

- max number of steps reached, for example `T = 3` or `T = 4`,
- the score no longer improves,
- the feature leaves supported territory,
- the gradient becomes too small,
- the realized move becomes negligible.

This iterative version should be interpreted differently from the one-step version:

- one-step = local causal probe,
- iterative = cumulative local control test.

## Why This Improves on the Binary Good/Bad Target

The score-based objective has several advantages.

### 1. It uses all available information

Two points that both fall inside `good` are no longer treated as identical.
If one point has much larger `p_good / p_bad`, the score reflects that.

### 2. It avoids large boundary-crossing jumps

The region-entry method can force a large move just to cross a boundary.
The score-based method instead asks:

```text
can we improve the local evidence for a correct trace?
```

### 3. It fits naturally with capped local moves

The update is defined directly by a local ascent direction and a perturbation budget.
This is cleaner than mixing:

- a distant target value,
- a step fraction,
- and a later cap.

### 4. It behaves better when distributions overlap

When correct and incorrect feature distributions overlap strongly, a hard region partition is often too coarse.
A smooth score remains meaningful in those situations.

## Recommended Default Variant

If this design is implemented, the recommended default is:

- support masking retained,
- smoothed log-density-ratio as the score,
- one capped step,
- relative hidden-state budget:

```text
tau = 0.005
```

- no multi-step loop by default,
- iterative mode as a separate experimental variant.

## Summary

The current region-based intervention can be summarized as:

```text
move toward the nearest point labeled good
```

The proposed score-based intervention replaces that with:

```text
take a capped local step that increases the supported smoothed log-density-ratio
```

That is a cleaner fit to what we already observed empirically:

- directionality matters,
- large jumps are often harmful,
- and small local improvements are more trustworthy than forced region crossings.
