from statsmodels.stats.power import TTestIndPower
import scipy.stats as st
import math
analysis = TTestIndPower()

def calc_z_score_and_p_value_given_samples(samples, true_mean, is_one_sided=False):
    x, s = samples.h.mean(), samples.h.std()
    return calc_z_score_and_p_value(x, s, true_mean, len(samples), is_one_sided=is_one_sided)    
    

def calc_z_score_and_p_value(sample_mean, sample_std, assumed_mean, n_samples, is_one_sided=False):
    se = sample_std / math.sqrt(n_samples)
    z_score = (sample_mean - assumed_mean) / se
    return z_score, calc_p_value_from_z_score(z_score, is_one_sided)


def calc_p_value_from_z_score(z_score, is_one_sided=False):
    p_value = st.norm.sf(abs(z_score))
    return p_value if is_one_sided else 2 * p_value
    

def calc_power(p_1, p_2, n_samples, alpha=0.05):
    s1 = math.sqrt(p_1*(1-p_1))
    standard_effect_size = (p_1 - p_2) / s1
    return analysis.power(effect_size=standard_effect_size, nobs1=n_samples, alpha=alpha, ratio=1)


def calc_needed_samples(curr_conv_rate, relative_increase, alpha, power):
    curr_conv_rate_std = math.sqrt(curr_conv_rate * (1 - curr_conv_rate))
    std_effect_size = (curr_conv_rate * (1 + relative_increase) - curr_conv_rate) / curr_conv_rate_std    
    return analysis.solve_power(effect_size=std_effect_size, alpha=alpha, power=power , ratio=1, nobs1=None, alternative='two-sided')