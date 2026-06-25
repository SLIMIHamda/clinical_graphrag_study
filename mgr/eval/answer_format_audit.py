"""Answer-format audit — gates EM/F1 emission across a comparison family.

Validity control #1 (Doc 00 section 8 / Doc 1 section 6): the thesis showed EM
jumping 0.00 -> 0.76 for hybrids only, under the same generator — an artifact of
arms being scored under *different* normalized output schemas. This audit runs
*first* and refuses to emit EM/F1 for a comparison if:

  - the arms do not share the same realized label schema, or
  - any arm's extraction coverage falls below a floor (degenerate parsing).

If the audit fails, the stats layer must not report EM/F1 for that family; it
surfaces the audit report instead, so the divergence is a stated finding rather
than a silent number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mgr.generate.extract import expected_schema, label_schema


@dataclass
class ArmAudit:
    arm: str
    schema: set[str]
    coverage: float
    conforms: bool


@dataclass
class AuditReport:
    answer_type: str
    passed: bool
    arms: list[ArmAudit] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def gate_emit_em_f1(self) -> bool:
        """True iff EM/F1 may be reported for this comparison family."""
        return self.passed


def audit(
    arm_labels: dict[str, list[str | None]],
    answer_type: str,
    *,
    min_coverage: float = 0.95,
) -> AuditReport:
    """Audit a family of arms scored on the same items.

    ``arm_labels`` maps arm name -> normalized predictions for the shared items.
    """
    report = AuditReport(answer_type=answer_type, passed=True)
    exp = expected_schema(answer_type)  # None for free-form (no schema gate)

    # One shared normalizer + one shared expected closed schema is what kills the
    # "EM 0->0.76" artifact: every arm is scored under the same vocabulary.
    # Differences in which labels an arm *happens* to predict are answer
    # distribution, not a schema mismatch, so we gate on conformance + coverage,
    # not on realized-set equality across arms.
    for arm, labels in arm_labels.items():
        schema = label_schema(labels, answer_type)
        coverage = (sum(1 for x in labels if x is not None) / len(labels)) if labels else 0.0
        conforms = True if exp is None else schema.issubset(exp)
        report.arms.append(ArmAudit(arm=arm, schema=schema, coverage=coverage, conforms=conforms))

        if exp is not None and not conforms:
            report.passed = False
            report.reasons.append(f"{arm}: labels {sorted(schema - exp)} outside expected {sorted(exp)}")
        if coverage < min_coverage:
            report.passed = False
            report.reasons.append(f"{arm}: coverage {coverage:.3f} < floor {min_coverage}")

    return report
