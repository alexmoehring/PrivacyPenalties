
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import tqdm
import matplotlib.pyplot as plt
import ResearchTools


def power(n, majority_share, beta_0_maj, beta_0_min, pclick_with_alg, treated_share):
    pvals = []
    n = int(n)
    for b in range(200):
        df = pd.DataFrame(index=range(n))
        df['d'] = pd.Series(np.random.rand(n) < treated_share).astype(int)
        df['majority'] = pd.Series(np.random.rand(n) < majority_share).astype(int)
        df['y0_maj'] = pd.Series(np.random.rand(n) < beta_0_maj).astype(int)
        df['y0_min'] = pd.Series(np.random.rand(n) < beta_0_min).astype(int)
        df['y1'] = pd.Series(np.random.rand(n) < pclick_with_alg).astype(int)
        df['y0'] = df.y0_maj * df.majority + (1-df.majority) * df.y0_min
        df['y'] = df.d * df.y1 + (1 - df.d) * df.y0

        # now estimate model
        mod = smf.ols('y ~ majority*d', data=df).fit()
        pval = mod.pvalues['majority:d']
        pvals.append(pval)
    p_reject = np.mean([el < 0.05 for el in pvals])
    return p_reject


def mde(n, tau_maj, pclick_with_alg, treated_share, majority_share, required_power):
    for el in np.linspace(0, 0.2, 21):
        if el > pclick_with_alg:
            continue
        tau_min_marginal = el
        beta_0_maj = pclick_with_alg - tau_maj
        beta_0_min = pclick_with_alg - tau_maj - tau_min_marginal
        p = power(n=n, majority_share=majority_share, beta_0_maj=beta_0_maj, beta_0_min=beta_0_min,
                  pclick_with_alg=pclick_with_alg, treated_share=treated_share)
        if p > required_power:
            return el
    return np.nan


def main():
    treated_share = 0.5

    pclick_with_algs = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    pclick_with_algs = [0.3]
    tau_maj = 0.02
    majority_share = 0.7
    # power(n, majority_share, beta_0_maj, beta_0_min, pclick_with_alg, treated_share=treated_share)

    mdes = pd.DataFrame(index=np.linspace(1000, 10000, 10), columns=pclick_with_algs)
    for n in tqdm.tqdm(mdes.index):
        for pclick_with_alg in pclick_with_algs:
            mdes.loc[n, pclick_with_alg] = mde(n, tau_maj, pclick_with_alg, treated_share, majority_share=majority_share, required_power=0.8)
    fig, ax = plt.subplots()
    mdes.plot(ax=ax)
    plt.xlabel('Number of subjects')
    plt.ylabel('Minimum detectable effect at 80% power')
    ResearchTools.ChartTools.save_show_plot(fig=fig, fn='./temp/power.pdf', show_graph=False, pickle_fig=False)


if __name__ == '__main__':
    main()
