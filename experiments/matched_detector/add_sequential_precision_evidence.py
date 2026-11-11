from pathlib import Path
from docx import Document
ROOT=Path('/mnt/data/BRCP_Revision_2026-11-11_Followup')
s=(ROOT/'main.tex').read_text()
key=r'\paragraph{Reference-algorithm and checkpoint provenance audit.}'
assert key in s
add=r'''
\paragraph{Sequential calibration-stability follow-up (exploratory).}
The previous 90-development-seed experiment failed to transfer its
false-alarm budget to the first test sample. We therefore ran a
\emph{sequential exploratory follow-up}, specifying its new seeds and
threshold grid before inspecting each new outcome but \emph{after}
observing the first failure. A larger calibration sample (1,200
new development seeds) selected $h_{\rm score}=5$ and
$h_{\rm data}=13$; on 900 disjoint new stationary-AR(1) null seeds
these yielded $0.5344$ and $0.4822$ alarms per 800 steps, a difference
of $0.0522$ (paired-bootstrap 95\% interval $[-0.018,0.122]$), within
that follow-up's stated tolerance of $0.15$. Without any further
threshold adjustment, a third independent set of 1,000 null seeds
yielded $0.524$ and $0.492$ alarms per 800 steps (gap $0.032$;
bootstrap difference interval $[-0.027,0.093]$). Thus larger
\emph{development} samples improved calibration stability for this
particular exactly stationary synthetic null. These follow-up choices
were motivated by inspecting the earlier result and are \emph{not}
an a priori registration of the entire sequence of studies.

On the third set of independently seeded persistent shifts, the
120-step detection proportions were $0.984$ (score) versus $0.977$
(data) for innovation-variance changes, with a paired hit-rate
difference of $0.007$ (bootstrap interval $[-0.005,0.019]$);
the conditional median detection delays were 3 versus 17 steps.
For persistent level changes the hit rates were $0.687$ versus
$0.974$, with a difference of $-0.287$ (bootstrap interval
$[-0.316,-0.258]$) and conditional median delays of 44 versus
17 steps. Delay medians are conditional on an alarm, so they should
not be interpreted independently of detection probability.
These results qualify the central scientific intuition: monitoring
conformity scores may respond rapidly to variance-driven competence
shifts, but can be less sensitive to an observable level change.
It is not universally superior to monitoring observations. This
sequential series is \emph{detector-only} and does not establish
matched operating characteristics on the real or Chronos score
streams. The original failure, larger-calibration results, frozen
follow-up thresholds, raw new-seed results and both bootstrap files
are retained, including the unfavorable level-shift outcomes.

'''
s=s.replace(key,add+key,1)
# Correct conclusion to give both studies full context
old='''The independent-seed synthetic detector null check also failed its prespecified held-out alarm-rate matching tolerance; consequently no general superiority of monitoring spaces is inferred from the unmatched hit rates.'''
new='''The initial independent-seed synthetic detector null check failed its prespecified held-out alarm-rate matching tolerance; a sequential follow-up using substantially more development data matched observed null alarm frequencies on separate new AR(1) test seeds, but showed strong level-shift dependence and does not establish general monitoring-space superiority.'''
assert old in s
s=s.replace(old,new)
(ROOT/'main.tex').write_text(s)

p=ROOT/'BRCP_Point_by_Point_Response_FURTHER_AUDIT_PRE_SUBMISSION.docx'
d=Document(p)
d.add_heading('Sequential larger-development detector stability check — 11 November 2026',level=1)
d.add_paragraph('Status: EXPLORATORY FOLLOW-UP INITIATED AFTER THE FIRST FROZEN AUDIT DID NOT MATCH TEST FALSE-ALARM RATES. This is not an a priori registration of the entire sequence.')
d.add_heading('Reviewer 1 / Reviewer 3 — improved synthetic null calibration (new independent seeds)',level=2)
d.add_paragraph('A new expanded development sample of 1,200 synthetic stationary Gaussian AR(1) null streams selected h_score=5 and h_data=13 from a separately fixed grid. We applied these thresholds without retuning to 900 fresh stationary null streams, obtaining 0.5344 versus 0.4822 alarms per 800 steps (difference 0.0522; paired bootstrap 95% interval [-0.018, 0.122]). On a third independent set of 1,000 null streams with the same fixed thresholds, observed rates were 0.524 versus 0.492 per 800 (difference 0.032; bootstrap interval [-0.027, 0.093]). This supports calibration stability under one specified synthetic stationary null when ample development data are available. It does not prove equivalence on the historical / published-checkpoint test series.')
d.add_heading('Reviewer 1 — break-family heterogeneity at near-matched synthetic null rates',level=2)
d.add_paragraph('The frozen thresholds from the 1,200-seed development set were also assessed on persistent changes generated from a third, independent set of 1,000 seeds. Within 120 post-break steps, score-space versus data-space detector hit rates were 0.984 versus 0.977 for a variance change (paired difference 0.007, 95% bootstrap interval [-0.005, 0.019]); conditional median delays were 3 versus 17 steps. For a level change the corresponding hit rates were 0.687 versus 0.974 (difference -0.287, interval [-0.316, -0.258]); median delays conditional on detection were 44 versus 17. Thus the monitoring-space advantage is not universal: early detection under variance shifts comes with much weaker detection of persistent level shifts. We retain and report the first failed null-matching experiment, as well as both later matched-null checks, rather than silently selecting favorable results.')
d.add_paragraph('Scientific gate: synthetic AR(1) detector-only null matching is supported in the expanded exploratory follow-up. Full published TSFM/real-data matched-budget evidence, source-equivalent AgACI/Conformal PID implementations, and exact pretraining-corpus disjointness remain unresolved. No novel Chronos GPU forward pass was run in this addendum.')
d.save(p)
print('Sequential evidence appended')
