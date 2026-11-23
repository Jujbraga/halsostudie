import numpy as np
import pandas as pd
from scipy import stats

def simulate_disease_proportion(data):
    rng = np.random.default_rng(15)

    with_disease = int(data.sum())
    mean_disease = data.mean()

    sample = rng.binomial(n=1, p=mean_disease, size=1000)
    se_disease = np.sqrt(mean_disease * (1 - mean_disease) / 1000)

    comparison = pd.DataFrame({
        "Antal personer": [len(data), len(sample)],
        "Antal sjuka": [with_disease, int(sample.sum())],
        "Andel sjuka": [mean_disease, sample.mean()],
    }, index=["Dataset", "Simulering"])

    text = f"""
    Totalt antal personer med sjukdomen: {with_disease} av {data.count()}
    Andelen personer med sjukdomen: {mean_disease} ({mean_disease*100}%)
    Standardfel (SE): {se_disease:.3%}
    """
    
    show_comparison = comparison.style.format({"Andel sjuka": "{:.3%}"})

    return (text, show_comparison)

def calculate_ci_t(data):
    bp = data.values
    n_bp = bp.size
    mean_bp = data.mean()
    sd_bp = bp.std(ddof=1)
    se_bp = sd_bp / np.sqrt(n_bp)

    t_crit = stats.t.ppf(0.975, df=n_bp-1)

    ci_low_t = mean_bp - t_crit * se_bp
    ci_high_t = mean_bp + t_crit * se_bp

    print(f"n = {n_bp}, medel = {mean_bp:.3f} mmHg, SD = {sd_bp:.3f}, SE = {se_bp:.3f}")

    return (ci_low_t, ci_high_t)

def calculate_ci_boot(data):
    rng_boot = np.random.default_rng(33)

    bp = data.values
    n_bp = bp.size
    n_boot = 10_000
    boot_means = []
    for _ in range(n_boot):
        boot_sample = rng_boot.choice(bp, size=n_bp, replace=True)
        boot_means.append(boot_sample.mean())
    ci_low_boot, ci_high_boot = np.percentile(boot_means, [2.5, 97.5])

    return (ci_low_boot, ci_high_boot)

def one_t_welch(a, b):
    t_stat, p_one = stats.ttest_ind(
    b, a,
    equal_var=False,
    alternative="greater")
    return (t_stat, p_one)

def calculate_hypothesis(a, b):
    n_b, n_a = len(b), len(a)
    mean_b, mean_a = b.mean(), a.mean()
    sd_b, sd_a = b.std(ddof=1), a.std(ddof=1)
    diff = mean_b - mean_a

    t_stat, p_one = one_t_welch(a, b)

    alpha = 0.05

    group = pd.DataFrame({
        "n": [n_b, n_a],
        "Medel (mmHg)": [mean_b, mean_a],
        "SD": [sd_b, sd_a],
    }, index=["Rökare", "Icke-rökare"]).round(3)

    result = pd.Series({
        "Skillnad i medelvärde": f"{diff:.3f} mmHg",
        "t-värde": f"{t_stat:.3f}",
        "p-värde (ensidigt)": f"{p_one:.4f}",
        "Beslut": "Förkasta H0" if p_one < alpha else "Förkasta inte H0",
    }, name="Rökare > Icke-rökare")

    return (group, result)

def simulate_power(a, b, diff, rng, n_sim=10_000, alpha=0.05):
    n_a, n_b = len(a), len(b)
    mean_a = a.mean()
    sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
    significant = 0

    for _ in range(n_sim):
        sample_a = rng.normal(mean_a, sd_a, n_a)
        sample_b = rng.normal(mean_a + diff, sd_b, n_b)

        # Apply the same Welch's one-sided t-test as hypotestest
        _, p_value = one_t_welch(sample_a, sample_b)

        if p_value < alpha:
            significant += 1

    return significant / n_sim