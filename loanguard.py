"""
LoanGuard: AI for Safer Borrowing
==================================
XGBoost + Isotonic Calibration risk model for loan analysis.

Requirements:
    pip install xgboost scikit-learn numpy pandas

Usage:
    # Train the model (creates risk_model.pkl and rep_model.pkl)
    python loanguard.py --train

    # Analyze a loan interactively
    python loanguard.py --analyze

    # Analyze a loan via command-line arguments
    python loanguard.py --loan-amount 2000 --apr 34 --term 24 --fee 150

    # Run demo with example loans
    python loanguard.py --demo
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

FEATURES = [
    "loan_amount",
    "apr",
    "term_months",
    "origination_fee",
    "origination_fee_pct",
    "has_prepayment_penalty",
    "has_variable_rate",
    "late_fee",
    "monthly_payment",
    "interest_to_principal_ratio",
    "debt_burden_score",
]

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
REP_MODEL_PATH  = os.path.join(MODEL_DIR, "rep_model.pkl")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_training_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic loan data for training."""
    rng = np.random.default_rng(seed)

    loan_amount            = rng.uniform(500, 50_000, n)
    apr                    = rng.uniform(4, 80, n)
    term_months            = rng.choice([12, 24, 36, 48, 60, 72, 84], n).astype(float)
    origination_fee_pct    = rng.uniform(0, 10, n)
    origination_fee        = origination_fee_pct * loan_amount / 100
    has_prepayment_penalty = rng.choice([0, 1], n, p=[0.7, 0.3]).astype(float)
    has_variable_rate      = rng.choice([0, 1], n, p=[0.6, 0.4]).astype(float)
    late_fee               = rng.uniform(0, 50, n)

    monthly_rate   = apr / 100 / 12
    monthly_payment = np.where(
        monthly_rate > 0,
        loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months)),
        loan_amount / term_months,
    )
    total_repayment            = monthly_payment * term_months + origination_fee
    interest_to_principal_ratio = (monthly_payment * term_months - loan_amount) / loan_amount
    debt_burden_score          = monthly_payment / (loan_amount / 12)

    # Risk label: high-risk if multiple bad factors co-occur
    risk_raw = (
        (apr > 25).astype(int) * 2
        + (apr > 40).astype(int) * 2
        + (origination_fee_pct > 5).astype(int) * 1.5
        + (term_months > 48).astype(int) * 1.0
        + has_prepayment_penalty * 1.0
        + has_variable_rate * 0.5
        + (interest_to_principal_ratio > 0.5).astype(int) * 1.5
        + (interest_to_principal_ratio > 1.0).astype(int) * 2.0
        + rng.normal(0, 0.5, n)
    )
    is_risky = (risk_raw > 4).astype(int)

    return pd.DataFrame(
        {
            "loan_amount":                loan_amount,
            "apr":                        apr,
            "term_months":                term_months,
            "origination_fee":            origination_fee,
            "origination_fee_pct":        origination_fee_pct,
            "has_prepayment_penalty":     has_prepayment_penalty,
            "has_variable_rate":          has_variable_rate,
            "late_fee":                   late_fee,
            "monthly_payment":            monthly_payment,
            "total_repayment":            total_repayment,
            "interest_to_principal_ratio": interest_to_principal_ratio,
            "debt_burden_score":          debt_burden_score,
            "is_risky":                   is_risky,
        }
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_samples: int = 5000, seed: int = 42, verbose: bool = True) -> dict:
    """
    Train the risk classifier and repayment regressor.
    Saves models to risk_model.pkl and rep_model.pkl.
    Returns a dict of evaluation metrics.
    """
    if verbose:
        print("Generating training data...")
    df = generate_training_data(n_samples, seed)

    X        = df[FEATURES]
    y_risk   = df["is_risky"]
    y_repay  = df["total_repayment"]

    X_train, X_test, yr_train, yr_test, yrep_train, yrep_test = train_test_split(
        X, y_risk, y_repay, test_size=0.2, random_state=seed, stratify=y_risk
    )

    # --- Risk classifier ---
    if verbose:
        print("Training XGBoost risk classifier with isotonic calibration...")
    pos_weight = (yr_train == 0).sum() / (yr_train == 1).sum()
    xgb_base = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=seed,
        verbosity=0,
    )
    risk_model = CalibratedClassifierCV(xgb_base, method="isotonic", cv=3)
    risk_model.fit(X_train, yr_train)

    # --- Repayment regressor ---
    if verbose:
        print("Training XGBoost repayment regressor...")
    rep_model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
    )
    rep_model.fit(X_train, yrep_train)

    # --- Evaluation ---
    y_prob   = risk_model.predict_proba(X_test)[:, 1]
    auc      = roc_auc_score(yr_test, y_prob)

    # Recall at fixed 5% FPR
    best_recall = 0.0
    n_neg = (yr_test == 0).sum()
    for t in np.linspace(0, 1, 500):
        y_pred = (y_prob >= t).astype(int)
        fpr = ((y_pred == 1) & (yr_test == 0)).sum() / n_neg
        if fpr <= 0.05:
            recall = ((y_pred == 1) & (yr_test == 1)).sum() / (yr_test == 1).sum()
            best_recall = max(best_recall, recall)

    rep_pred = rep_model.predict(X_test)
    mape     = float(np.mean(np.abs((yrep_test - rep_pred) / yrep_test)) * 100)

    metrics = {
        "auc":               round(float(auc), 4),
        "recall_at_5pct_fpr": round(float(best_recall), 4),
        "repayment_mape_pct": round(mape, 2),
    }

    # --- Persist ---
    with open(RISK_MODEL_PATH, "wb") as f:
        pickle.dump(risk_model, f)
    with open(REP_MODEL_PATH, "wb") as f:
        pickle.dump(rep_model, f)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print("\n── Model Performance ──────────────────────")
        print(f"  AUC:                 {metrics['auc']}  (target ≥ 0.88)")
        print(f"  Recall @ 5% FPR:     {metrics['recall_at_5pct_fpr']*100:.1f}%  (target ≥ 90%)")
        print(f"  Repayment MAPE:      {metrics['repayment_mape_pct']}%  (target < 10%)")
        print(f"\nModels saved → {RISK_MODEL_PATH}")
        print(f"               {REP_MODEL_PATH}")

    return metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_models():
    """Load saved models, training first if not found."""
    if not os.path.exists(RISK_MODEL_PATH) or not os.path.exists(REP_MODEL_PATH):
        print("Models not found — training now...\n")
        train()
    with open(RISK_MODEL_PATH, "rb") as f:
        risk_model = pickle.load(f)
    with open(REP_MODEL_PATH, "rb") as f:
        rep_model = pickle.load(f)
    return risk_model, rep_model


def _derive_features(
    loan_amount: float,
    apr: float,
    term_months: int,
    origination_fee: float,
    late_fee: float,
):
    """Compute derived features from raw loan inputs."""
    monthly_rate = apr / 100 / 12
    if monthly_rate > 0:
        monthly_payment = (
            loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))
        )
    else:
        monthly_payment = loan_amount / term_months

    interest_to_principal = (monthly_payment * term_months - loan_amount) / loan_amount
    debt_burden_score     = monthly_payment / (loan_amount / 12)
    origination_fee_pct   = (origination_fee / loan_amount) * 100
    return monthly_payment, interest_to_principal, debt_burden_score, origination_fee_pct


def predict(
    loan_amount: float,
    apr: float,
    term_months: int,
    origination_fee: float = 0.0,
    has_prepayment_penalty: bool = False,
    has_variable_rate: bool = False,
    late_fee: float = 0.0,
    _models=None,
) -> dict:
    """
    Analyze a single loan and return a risk assessment.

    Parameters
    ----------
    loan_amount             : Total loan principal ($)
    apr                     : Annual Percentage Rate (%)
    term_months             : Repayment duration in months
    origination_fee         : Upfront origination fee ($)
    has_prepayment_penalty  : True if loan penalizes early repayment
    has_variable_rate       : True if interest rate can change
    late_fee                : Per-event late payment fee ($)

    Returns
    -------
    dict with keys:
        risk_score          : 0–100 (higher = riskier)
        risk_category       : 'Low Risk' | 'Medium Risk' | 'High Risk'
        risk_probability    : Calibrated probability (0.0–1.0)
        monthly_payment     : Estimated monthly payment ($)
        total_repayment     : Estimated total amount repaid ($)
        total_interest      : Total interest + fees paid ($)
        reasons             : List of plain-language risk explanations
    """
    risk_model, rep_model = _models or _load_models()

    mp, ipr, dbs, fee_pct = _derive_features(
        loan_amount, apr, term_months, origination_fee, late_fee
    )

    X = np.array(
        [[
            loan_amount,
            apr,
            term_months,
            origination_fee,
            fee_pct,
            float(has_prepayment_penalty),
            float(has_variable_rate),
            late_fee,
            mp,
            ipr,
            dbs,
        ]]
    )

    risk_prob   = float(risk_model.predict_proba(X)[0][1])
    risk_score  = int(round(risk_prob * 100))
    total_repay = float(rep_model.predict(X)[0])

    if risk_score >= 70:
        category = "High Risk"
    elif risk_score >= 40:
        category = "Medium Risk"
    else:
        category = "Low Risk"

    reasons = []
    if apr > 25:
        reasons.append(
            f"High APR of {apr}% — well above typical personal loan rates (6–15%)"
        )
    if apr > 40:
        reasons.append(
            "Extremely high interest rate — characteristic of predatory lending"
        )
    if fee_pct > 5:
        reasons.append(
            f"Origination fee is {fee_pct:.1f}% of the loan amount, "
            "significantly increasing effective borrowing cost"
        )
    if term_months > 48:
        reasons.append(
            f"{term_months}-month term maximises total interest paid to lender"
        )
    if has_prepayment_penalty:
        reasons.append(
            "Prepayment penalty prevents saving on interest by paying off early"
        )
    if has_variable_rate:
        reasons.append(
            "Variable rate introduces future payment uncertainty"
        )
    if ipr > 0.5:
        reasons.append(
            f"Total interest ({ipr * 100:.0f}% of principal) represents a heavy interest burden"
        )
    if late_fee > 30:
        reasons.append(
            f"High late fee (${late_fee:.0f}) adds significant risk for missed payments"
        )
    if not reasons:
        reasons.append(
            "Loan terms appear within typical market ranges — review full contract carefully"
        )

    return {
        "risk_score":       risk_score,
        "risk_category":    category,
        "risk_probability": round(risk_prob, 4),
        "monthly_payment":  round(mp, 2),
        "total_repayment":  round(total_repay, 2),
        "total_interest":   round(total_repay - loan_amount, 2),
        "reasons":          reasons,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_result(result: dict, loan_amount: float, apr: float, term: int, fee: float):
    score    = result["risk_score"]
    cat      = result["risk_category"]
    stars    = "█" * (score // 10) + "░" * (10 - score // 10)

    print("\n" + "═" * 52)
    print("  LoanGuard Risk Assessment")
    print("═" * 52)
    print(f"  Loan:       ${loan_amount:,.0f}  |  APR: {apr}%  |  Term: {term} mo")
    print(f"  Origin fee: ${fee:,.0f}")
    print("─" * 52)
    print(f"  Risk Score:   {score}/100  [{stars}]")
    print(f"  Category:     {cat}")
    print(f"  Probability:  {result['risk_probability']:.1%}")
    print("─" * 52)
    print(f"  Monthly Payment:   ${result['monthly_payment']:>8,.2f}")
    print(f"  Total Repayment:   ${result['total_repayment']:>8,.2f}")
    print(f"  Total Interest:    ${result['total_interest']:>8,.2f}")
    print("─" * 52)
    print("  Risk Factors:")
    for r in result["reasons"]:
        print(f"    • {r}")
    print("═" * 52 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def interactive_analyze():
    """Prompt the user for loan details and print a risk assessment."""
    print("\n── LoanGuard Interactive Analyzer ──────────")
    try:
        loan_amount  = float(input("  Loan amount ($):           ") or 2000)
        apr          = float(input("  APR (%):                   ") or 10)
        term         = int(input("  Term (months):             ") or 24)
        fee          = float(input("  Origination fee ($):       ") or 0)
        late_fee     = float(input("  Late fee ($):              ") or 25)
        prepay_in    = input("  Prepayment penalty? (y/n): ").strip().lower()
        variable_in  = input("  Variable rate?     (y/n): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    result = predict(
        loan_amount=loan_amount,
        apr=apr,
        term_months=term,
        origination_fee=fee,
        has_prepayment_penalty=(prepay_in == "y"),
        has_variable_rate=(variable_in == "y"),
        late_fee=late_fee,
    )
    print_result(result, loan_amount, apr, term, fee)


def run_demo():
    """Print assessments for a set of illustrative loan examples."""
    examples = [
        dict(label="Good loan",       loan_amount=10_000, apr=8,  term_months=36, origination_fee=0,   has_prepayment_penalty=False, has_variable_rate=False, late_fee=15),
        dict(label="Document example",loan_amount=2_000,  apr=34, term_months=24, origination_fee=150, has_prepayment_penalty=False, has_variable_rate=False, late_fee=25),
        dict(label="Borderline loan", loan_amount=5_000,  apr=22, term_months=48, origination_fee=200, has_prepayment_penalty=False, has_variable_rate=True,  late_fee=25),
        dict(label="High-risk loan",  loan_amount=3_000,  apr=55, term_months=60, origination_fee=450, has_prepayment_penalty=True,  has_variable_rate=True,  late_fee=50),
        dict(label="Predatory loan",  loan_amount=1_500,  apr=79, term_months=84, origination_fee=300, has_prepayment_penalty=True,  has_variable_rate=True,  late_fee=50),
    ]

    models = _load_models()
    for ex in examples:
        label = ex.pop("label")
        print(f"\n{'─'*52}\n  Demo: {label}")
        result = predict(**ex, _models=models)
        print_result(result, ex["loan_amount"], ex["apr"], ex["term_months"], ex["origination_fee"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoanGuard — AI loan risk analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train",       action="store_true", help="Train and save models")
    parser.add_argument("--analyze",     action="store_true", help="Interactive loan analyzer")
    parser.add_argument("--demo",        action="store_true", help="Run demo with example loans")
    parser.add_argument("--loan-amount", type=float, help="Loan principal ($)")
    parser.add_argument("--apr",         type=float, help="Annual Percentage Rate (%%)")
    parser.add_argument("--term",        type=int,   help="Term in months")
    parser.add_argument("--fee",         type=float, default=0,    help="Origination fee ($)")
    parser.add_argument("--late-fee",    type=float, default=25,   help="Late fee ($)")
    parser.add_argument("--prepay",      action="store_true",      help="Has prepayment penalty")
    parser.add_argument("--variable",    action="store_true",      help="Has variable rate")

    args = parser.parse_args()

    if args.train:
        train()
    elif args.demo:
        run_demo()
    elif args.analyze:
        interactive_analyze()
    elif args.loan_amount and args.apr and args.term:
        result = predict(
            loan_amount=args.loan_amount,
            apr=args.apr,
            term_months=args.term,
            origination_fee=args.fee,
            has_prepayment_penalty=args.prepay,
            has_variable_rate=args.variable,
            late_fee=args.late_fee,
        )
        print_result(result, args.loan_amount, args.apr, args.term, args.fee)
    else:
        parser.print_help()
        print("\nQuick start:  python loanguard.py --demo\n")


if __name__ == "__main__":
    main()