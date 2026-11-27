import matplotlib.pyplot as plt
import numpy as np

def boxplot_weight_gender(df):
    """Plot a boxplot of weight for each sex"""
    fig, ax = plt.subplots(figsize=(8,4))

    medianprops = {"color": "black", "linewidth": 2}

    df.boxplot(
        column="weight", 
        by="sex", 
        ax=ax, 
        medianprops=medianprops,
        flierprops={"marker": "o", "markerfacecolor": "none", "alpha": 0.6},
        patch_artist=True, 
    )

    colors = ["#e7a1c0", "#8fb8de"]

    for patch, color in zip(ax.patches, colors):
        patch.set_facecolor(color)

    fig.suptitle("Vikt per kön")
    ax.set_title("")
    ax.set_xlabel("Kön")
    ax.set_ylabel("Vikt (kg)")
    ax.grid(False, axis="x")
    plt.show()

def scatter_cholesterol(df):
    """Plot cholesterol vs age as a scatter plot with a trendline"""
    x = df["age"].to_numpy()
    y = df["cholesterol"].to_numpy()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(x, y, alpha=0.6)

    # Trendline
    k, m = np.polyfit(x, y, 1) 
    ix = np.argsort(x)
    ax.plot(x[ix], (k*x + m)[ix], color="purple")

    ax.set_title("Kolesterol vs ålder")
    ax.set_xlabel("Ålder (år)")
    ax.set_ylabel("Kolesterol (mmol/L)")
    ax.grid(True)
    plt.show()

def hist_bmi(df):
    """Plot a histogram of BMI with lines marking the underweight, normal, overweight and obesity limits"""
    fig, (ax) = plt.subplots(figsize=(8, 4))
    ax.hist(df["bmi"], bins=25, edgecolor="white")

    ax.axvline(18.5, color="blue", linestyle="--", label="Under/normalvikt")
    ax.axvline(25, color="orange", linestyle="--", label="Normal/övervikt")
    ax.axvline(30, color="red", linestyle="--", label="Övervikt/fetma")

    ax.set_title("Fördelning av BMI")
    ax.set_ylabel("Antal personer")
    plt.xlabel("BMI (kg/m²)")
    ax.grid(True, axis="y")
    plt.legend()
    plt.show()

def scatter_regression(x, y, interc, coef):
    """Plot blood pressure vs age as a scatter plot with the regression line given by interc and coef"""
    plt.figure(figsize=(6,4))
    plt.scatter(x[:,0], y, alpha=0.6)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = interc + coef * x_line
    plt.plot(x_line, y_line, color="black")
    plt.xlabel("Ålder (år)")
    plt.ylabel("Blodtryck (mmHg)")
    plt.title("Blodtryck vs ålder")
    plt.show()

def scatter_residuals(x, y):
    """Plot residuals (y) against predicted blood pressure (x) to check the fit of the regression"""
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6)
    plt.axhline(y=0, color="black", linestyle="--")
    plt.xlabel("Förutsagt blodtryck (mmHg)")
    plt.ylabel("Residualer (mmHg)")
    plt.title("Residualplot - multipel linjär regression")
    plt.show()