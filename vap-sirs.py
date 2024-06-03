from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from scipy.integrate import odeint
import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Define the SIR model
def vape_sir_model(y, t, beta_s, beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                   ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                   omega_m, gamma_s, gamma_m):

  S1s, S1m, S2s, S2m, Sns, Snm, Sds, Sdm, I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm, Rvm, Rvs, Rns, Rnm, Rds, Rdm, Vs, Vm = y

  I = Ins + Inm + Ids + Idm
  Iv = I1s + I1m + I2s + I2m

  # Susceptible
  dSdsdt = -(beta_s * I + beta_vs * Iv) * Sds + kappa_s * Rds
  dSdmdt = -(beta_m * I + beta_vm * Iv) * Sdm + kappa_m * Rdm

  dSnmdt = -(beta_m * I + beta_vm * Iv) * Snm + kappa_m * Rnm - ni_m * Snm
  dSnsdt = -(beta_s * I + beta_vs * Iv) * Sns + kappa_s * Rns - ni_s * Sns

  dS1sdt = ni_rs * (1 - alpha_s) * S2s + ni_s * (
      1 - alpha_s) * Sns - omega_s * S1s - (beta_s * I + beta_vs * Iv) * S1s
  dS1mdt = ni_rm * (1 - alpha_m) * S2m + ni_m * (
      1 - alpha_m) * Snm - omega_m * S1m - (beta_m * I + beta_vm * Iv) * S1m

  dS2sdt = -ni_rs * S2s + omega_s * Vs + omega_s * S1s - (
      beta_s * I + beta_vs * Iv) * S2s + kappa_s * Rvs
  dS2mdt = -ni_rm * S2m + omega_m * Vm + omega_m * S1m - (
      beta_m * I + beta_vm * Iv) * S2m + kappa_m * Rvm

  # Infected
  dInsdt = (beta_s * I + beta_vs * Iv) * Sns - gamma_s * Ins
  dInmdt = (beta_m * I + beta_vm * Iv) * Snm - gamma_m * Inm
  dIdsdt = (beta_s * I + beta_vs * Iv) * Sds - gamma_s * Ids
  dIdmdt = (beta_m * I + beta_vm * Iv) * Sdm - gamma_m * Idm

  dI1sdt = (beta_s * I + beta_vs * Iv) * S1s - gamma_s * I1s
  dI2sdt = (beta_s * I + beta_vs * Iv) * S2s - gamma_s * I2s
  dI1mdt = (beta_m * I + beta_vm * Iv) * S1m - gamma_m * I1m
  dI2mdt = (beta_m * I + beta_vm * Iv) * S2m - gamma_m * I2m

  # Recovered
  dRnsdt = gamma_s * Ins - kappa_s * Rns - ni_s * Rns
  dRnmdt = gamma_m * Inm - kappa_m * Rnm - ni_m * Rnm
  dRdsdt = gamma_s * Ids - kappa_s * Rds
  dRdmdt = gamma_m * Idm - kappa_m * Rdm
  dRvmdt = gamma_m * (I1m + I2m) - kappa_m * Rvm - ni_rm * Rvm
  dRvsdt = gamma_s * (I1s + I2s) - kappa_s * Rvs - ni_rs * Rvs

  # Vaccinated
  dVsdt = ni_s * alpha_s * Sns + ni_rs * alpha_s * S2s + ni_rs * Rvs + ni_s * Rns - omega_s * Vs
  dVmdt = ni_m * alpha_m * Snm + ni_rm * alpha_m * S2m + ni_rm * Rvm + ni_m * Rnm - omega_m * Vm

  return [
      dS1sdt, dS1mdt, dS2sdt, dS2mdt, dSnsdt, dSnmdt, dSdsdt, dSdmdt, dI1sdt,
      dI1mdt, dI2sdt, dI2mdt, dInsdt, dInmdt, dIdsdt, dIdmdt, dRvmdt, dRvsdt,
      dRnsdt, dRnmdt, dRdsdt, dRdmdt, dVsdt, dVmdt
  ]

def vape_sir_model_new_variant(y, t, beta_s, beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                   ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                   omega_m, gamma_s, gamma_m, new_variant_start, 
                   beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
                   kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
                   ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant):

    if t > new_variant_start:
        beta_s = beta_s_variant
        beta_m = beta_m_variant
        beta_vs = beta_vs_variant
        beta_vm = beta_vm_variant
        kappa_s = kappa_s_variant
        kappa_m = kappa_m_variant
        omega_s = omega_s_variant
        omega_m = omega_m_variant
        ni_s = ni_s_variant
        ni_m = ni_m_variant
        ni_rs = ni_rs_variant
        ni_rm = ni_rm_variant

    S1s, S1m, S2s, S2m, Sns, Snm, Sds, Sdm, I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm, Rvm, Rvs, Rns, Rnm, Rds, Rdm, Vs, Vm = y

    I = Ins + Inm + Ids + Idm
    Iv = I1s + I1m + I2s + I2m

    # Susceptible
    dSdsdt = -(beta_s * I + beta_vs * Iv) * Sds + kappa_s * Rds
    dSdmdt = -(beta_m * I + beta_vm * Iv) * Sdm + kappa_m * Rdm

    dSnmdt = -(beta_m * I + beta_vm * Iv) * Snm + kappa_m * Rnm - ni_m * Snm
    dSnsdt = -(beta_s * I + beta_vs * Iv) * Sns + kappa_s * Rns - ni_s * Sns

    dS1sdt = ni_rs * (1 - alpha_s) * S2s + ni_s * (1 - alpha_s) * Sns - omega_s * S1s - (beta_s * I + beta_vs * Iv) * S1s
    dS1mdt = ni_rm * (1 - alpha_m) * S2m + ni_m * (1 - alpha_m) * Snm - omega_m * S1m - (beta_m * I + beta_vm * Iv) * S1m

    dS2sdt = -ni_rs * S2s + omega_s * Vs + omega_s * S1s - (beta_s * I + beta_vs * Iv) * S2s + kappa_s * Rvs
    dS2mdt = -ni_rm * S2m + omega_m * Vm + omega_m * S1m - (beta_m * I + beta_vm * Iv) * S2m + kappa_m * Rvm

    # Infected
    dInsdt = (beta_s * I + beta_vs * Iv) * Sns - gamma_s * Ins
    dInmdt = (beta_m * I + beta_vm * Iv) * Snm - gamma_m * Inm
    dIdsdt = (beta_s * I + beta_vs * Iv) * Sds - gamma_s * Ids
    dIdmdt = (beta_m * I + beta_vm * Iv) * Sdm - gamma_m * Idm

    dI1sdt = (beta_s * I + beta_vs * Iv) * S1s - gamma_s * I1s
    dI2sdt = (beta_s * I + beta_vs * Iv) * S2s - gamma_s * I2s
    dI1mdt = (beta_m * I + beta_vm * Iv) * S1m - gamma_m * I1m
    dI2mdt = (beta_m * I + beta_vm * Iv) * S2m - gamma_m * I2m

    # Recovered
    dRnsdt = gamma_s * Ins - kappa_s * Rns - ni_s * Rns
    dRnmdt = gamma_m * Inm - kappa_m * Rnm - ni_m * Rnm
    dRdsdt = gamma_s * Ids - kappa_s * Rds
    dRdmdt = gamma_m * Idm - kappa_m * Rdm
    dRvmdt = gamma_m * (I1m + I2m) - kappa_m * Rvm - ni_rm * Rvm
    dRvsdt = gamma_s * (I1s + I2s) - kappa_s * Rvs - ni_rs * Rvs

    # Vaccinated
    dVsdt = ni_s * alpha_s * Sns + ni_rs * alpha_s * S2s + ni_rs * Rvs + ni_s * Rns - omega_s * Vs
    dVmdt = ni_m * alpha_m * Snm + ni_rm * alpha_m * S2m + ni_rm * Rvm + ni_m * Rnm - omega_m * Vm

    return [
        dS1sdt, dS1mdt, dS2sdt, dS2mdt, dSnsdt, dSnmdt, dSdsdt, dSdmdt, dI1sdt,
        dI1mdt, dI2sdt, dI2mdt, dInsdt, dInmdt, dIdsdt, dIdmdt, dRvmdt, dRvsdt,
        dRnsdt, dRnmdt, dRdsdt, dRdmdt, dVsdt, dVmdt
    ]

# Rectify negative values
def rectify_negative_values(results):
  results[results < 0] = 0
  return results

# Add noise to the data
def add_noise(data, noise_level):
  noise = np.random.normal(0, noise_level, data.shape)
  return data + noise

# Simulate SIR model
def simulate_sir_model(population_size, initial_conditions, beta_s, beta_m,
                       beta_vs, beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm,
                       ni_rs, alpha_m, alpha_s, omega_s, omega_m, gamma_s,
                       gamma_m, days, noise_level, should_add_noise=True):

  t = np.linspace(0, days, days)  # time points
  y0 = initial_conditions  # initial values

  result = odeint(vape_sir_model,
                  y0,
                  t,
                  args=(beta_s, beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                        ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                        omega_m, gamma_s, gamma_m))

  result = rectify_negative_values(result)
  if should_add_noise:
    result = add_noise(result, noise_level)

  return t, result

# Plot results
def plot_results(t, result, population_size, title):
    S1s, S1m, S2s, S2m, Sns, Snm, Sds, Sdm, I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm, Rvm, Rvs, Rns, Rnm, Rds, Rdm, Vs, Vm = result.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S1s, 'b-', label='Narażeni starsi bez odporności po szczepieniu')  # Solid blue
    plt.plot(t, S2s, 'g--', label='Narażeni starsi, którzy stracili odporność po szczepieniu')  # Dashed green
    plt.plot(t, Sns, 'r-.', label='Narażeni starsi, którzy chcą się zaszczepić')  # Dash-dot red
    plt.plot(t, Sds, 'c:', label='Narażeni starsi, którzy nie chcą się zaszczepić')  # Dotted cyan
    plt.plot(t, I1s, 'm-', label='Zarażeni starsi bez odporności po szczepieniu')  # Solid magenta
    plt.plot(t, I2s, 'y--', label='Zarażeni starsi, którzy stracili odporność po szczepieniu')  # Dashed yellow
    plt.plot(t, Ins, 'k-.', label='Zarażeni starsi, którzy chcą się zaszczepić')  # Dash-dot black
    plt.plot(t, Ids, 'b:', label='Zarażeni starsi, którzy nie chcą się zaszczepić')  # Dotted blue
    plt.plot(t, Rvs, 'g-', label='Ozdrowiali starsi, którzy się zaszczepili')  # Solid green
    plt.plot(t, Rns, 'r--', label='Ozdrowiali starsi, którzy chcą się zaszczepić')  # Dashed red
    plt.plot(t, Rds, 'c-.', label='Ozdrowiali starsi, którzy nie chcą się zaszczepić')  # Dash-dot cyan
    plt.plot(t, Vs, 'm:', label='Zaszczepieni starsi')  # Dotted magenta
    plt.ylabel('Population')
    plt.title(f'Model SIR - {title}')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(t, S1m, 'b-', label='Narażeni młodsi bez odporności po szczepieniu')  # Solid blue
    plt.plot(t, S2m, 'g--', label='Narażeni młodsi, którzy stracili odporność po szczepieniu')  # Dashed green
    plt.plot(t, Snm, 'r-.', label='Narażeni młodsi, którzy chcą się zaszczepić')  # Dash-dot red
    plt.plot(t, Sdm, 'c:', label='Narażeni młodsi, którzy nie chcą się zaszczepić')  # Dotted cyan
    plt.plot(t, I1m, 'm-', label='Zarażeni młodsi bez odporności po szczepieniu')  # Solid magenta
    plt.plot(t, I2m, 'y--', label='Zarażeni młodsi, którzy stracili odporność po szczepieniu')  # Dashed yellow
    plt.plot(t, Inm, 'k-.', label='Zarażeni młodsi, którzy chcą się zaszczepić')  # Dash-dot black
    plt.plot(t, Idm, 'b:', label='Zarażeni młodsi, którzy nie chcą się zaszczepić')  # Dotted blue
    plt.plot(t, Rvm, 'g-', label='Ozdrowiali młodsi, którzy się zaszczepili')  # Solid green
    plt.plot(t, Rnm, 'r--', label='Ozdrowiali młodsi, którzy chcą się zaszczepić')  # Dashed red
    plt.plot(t, Rdm, 'c-.', label='Ozdrowiali młodsi, którzy nie chcą się zaszczepić')  # Dash-dot cyan
    plt.plot(t, Vm, 'm:', label='Zaszczepieni młodsi')  # Dotted magenta

    plt.xlabel('Czas (dni)')
    plt.ylabel('Liczebność populacji')
    plt.title(f'Model SIR - podstawowy - {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot results for seasonal variation with horizontal lines and shading
def plot_results_seasonal(t, result, population_size, time_intervals):
    S1s, S1m, S2s, S2m, Sns, Snm, Sds, Sdm, I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm, Rvm, Rvs, Rns, Rnm, Rds, Rdm, Vs, Vm = result.T

    total_infected = (I1s + I1m + I2s + I2m + Ins + Inm + Ids + Idm) * population_size
    old_infected = (I1s + I2s + Ins + Ids) * population_size
    young_infected = (I1m + I2m + Inm + Idm) * population_size

    plt.figure(figsize=(12, 6))
    plt.plot(t, total_infected, label='Zarażeni łącznie', color='black')
    plt.plot(t, old_infected, label='Zarażeni starsi', color='red')
    plt.plot(t, young_infected, label='Zarażeni młodsi', color='blue')

    colors = ['lightgreen', 'yellow', 'orange', 'lightblue']
    season_labels = ['Wiosna', 'Lato', 'Jesień', 'Zima']
    season_handles = [Patch(color=colors[i], alpha=0.3, label=season_labels[i]) for i in range(4)]

    start = 0
    for i, interval in enumerate(time_intervals):
        end = start + interval
        plt.axvspan(start, end, color=colors[i % 4], alpha=0.3)
        plt.axvline(x=start, color='k', linestyle='--')
        start = end

    plt.xlabel('Czas (dni)')
    plt.ylabel('Liczebność populacji')
    plt.title(f'Model SIR - z sezonowością zakażeń')
    plt.legend(handles=season_handles + [plt.Line2D([0], [0], color='black', lw=2, label='Zarażeni łącznie'),
                                         plt.Line2D([0], [0], color='red', lw=2, label='Zarażeni starsi'),
                                         plt.Line2D([0], [0], color='blue', lw=2, label='Zarażeni młodsi')],
               loc='upper left')
    plt.grid(True)
    plt.show()

def simulate_sir_model_find_opt_beta(population_size, hospital_capacity, initial_conditions, beta_s, beta_m,
                       beta_vs, beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm,
                       ni_rs, alpha_m, alpha_s, omega_s, omega_m, gamma_s,
                       gamma_m, days, noise_level):
    t = np.linspace(0, days, days)  # time points
    y0 = initial_conditions  # initial values

    result = odeint(vape_sir_model, y0, t,
                    args=(beta_s, beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                          ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                          omega_m, gamma_s, gamma_m))

    result = rectify_negative_values(result)
    result = add_noise(result, noise_level)


    S1s, S1m, S2s, S2m, Sns, Snm, Sds, Sdm, I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm, Rvm, Rvs, Rns, Rnm, Rds, Rdm, Vs, Vm = result.T

    Im = I1m + I2m + Inm + Idm
    Is = I1s + I2s + Ins + Ids

    I = Im + Is

    return I

# Simulate SIR model with seasonal variation
def simulate_sir_model_seasonal_variation(
    population_size, initial_conditions, beta_s_values, beta_m_values,
    beta_vs_values, beta_vm_values, kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs,
    alpha_m, alpha_s, omega_s, omega_m, gamma_s, gamma_m, days, noise_level,
    time_intervals, should_add_noise=True):
  t_total = []
  result_total = []
  y0 = np.array(initial_conditions)

  for i, interval in enumerate(time_intervals):
    beta_s = beta_s_values[i]
    beta_m = beta_m_values[i]
    beta_vs = beta_vs_values[i]
    beta_vm = beta_vm_values[i]

    t = np.linspace(i * interval, (i + 1) * interval, interval)
    result = odeint(vape_sir_model,
                    y0,
                    t,
                    args=(beta_s, beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                          ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                          omega_m, gamma_s, gamma_m), atol=1e-12, rtol=1e-10)
    result = rectify_negative_values(result)
    if should_add_noise:
      result = add_noise(result, noise_level)
    y0 = result[-1]
    t_total.extend(t)
    result_total.append(result)

  result_total = np.vstack(result_total)
  return t_total, result_total

# Save synthetic data
def save_synthetic_data(filename, t_total, synthetic_data):
  df = pd.DataFrame(synthetic_data,
                    columns=[
                        'S1s', 'S1m', 'S2s', 'S2m', 'Sns', 'Snm', 'Sds', 'Sdm',
                        'I1s', 'I1m', 'I2s', 'I2m', 'Ins', 'Inm', 'Ids', 'Idm',
                        'Rvm', 'Rvs', 'Rns', 'Rnm', 'Rds', 'Rdm', 'Vs', 'Vm'
                    ])
  df['Time'] = t_total
  df.to_csv(filename, index=False)

# Load synthetic data
def load_synthetic_data(filename):
  df = pd.read_csv(filename)
  t_total = df['Time'].values
  synthetic_data = df.drop(columns=['Time']).values
  return t_total, synthetic_data

# Objective function for optimization
def objective_function(params, initial_conditions, t_total, observed_data,
                       population_size, time_intervals):
  beta_s_values, beta_m_values, beta_vs_values, beta_vm_values = np.split(
      params, 4)

  result_total = []
  y0 = np.array(initial_conditions)

  for i, interval in enumerate(time_intervals):
    beta_s = beta_s_values[i]
    beta_m = beta_m_values[i]
    beta_vs = beta_vs_values[i]
    beta_vm = beta_vm_values[i]

    t = np.linspace(i * interval, (i + 1) * interval, interval)
    result = odeint(vape_sir_model,
                    y0,
                    t,
                    args=(beta_s, beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                          ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                          omega_m, gamma_s, gamma_m))
    result = rectify_negative_values(result)
    y0 = result[-1]
    result_total.append(result)

  result_total = np.vstack(result_total)
  I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm = result_total[:, 8:16].T
  predicted_infected = (I1s + I1m + I2s + I2m + Ins + Inm + Ids +
                        Idm) * population_size

  error = observed_data - predicted_infected
  return np.sum(error**2)

# Run simulation
def run_simulation(population_size,
                   initial_conditions,
                   beta_s,
                   beta_m,
                   beta_vs,
                   beta_vm,
                   kappa_s,
                   kappa_m,
                   ni_m,
                   ni_s,
                   ni_rm,
                   ni_rs,
                   alpha_m,
                   alpha_s,
                   omega_s,
                   omega_m,
                   gamma_s,
                   gamma_m,
                   days,
                   noise_level,
                   title=''):
  t, result = simulate_sir_model(population_size, initial_conditions, beta_s,
                                 beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                                 ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s,
                                 omega_s, omega_m, gamma_s, gamma_m, days,
                                 noise_level)
  plot_results(t, result, population_size, title)



# Run seasonal simulation and save data
def run_seasonal_simulation(population_size,
                            initial_conditions,
                            beta_s_values,
                            beta_m_values,
                            beta_vs_values,
                            beta_vm_values,
                            kappa_s,
                            kappa_m,
                            ni_m,
                            ni_s,
                            ni_rm,
                            ni_rs,
                            alpha_m,
                            alpha_s,
                            omega_s,
                            omega_m,
                            gamma_s,
                            gamma_m,
                            days,
                            noise_level,
                            time_intervals,
                            filename=None):
  t_total, synthetic_data = simulate_sir_model_seasonal_variation(
      population_size, initial_conditions, beta_s_values, beta_m_values,
      beta_vs_values, beta_vm_values, kappa_s, kappa_m, ni_m, ni_s, ni_rm,
      ni_rs, alpha_m, alpha_s, omega_s, omega_m, gamma_s, gamma_m, days,
      noise_level, time_intervals)

  if filename:
    save_synthetic_data(filename, t_total, synthetic_data)

  plot_results_seasonal(t_total, synthetic_data, population_size, time_intervals)

  return t_total, synthetic_data

# Run optimization for seasonal variation
def run_optimization_seasonal(population_size, initial_conditions, time_intervals,
                     worse_initial_guess, filename):
  t_total, synthetic_data = load_synthetic_data(filename)

  total_infected_data = (
      synthetic_data[:, 8] + synthetic_data[:, 9] + synthetic_data[:, 10] +
      synthetic_data[:, 11] + synthetic_data[:, 12] + synthetic_data[:, 13] +
      synthetic_data[:, 14] + synthetic_data[:, 15]) * population_size

  result = minimize(objective_function,
                    worse_initial_guess,
                    args=(initial_conditions, t_total, total_infected_data,
                          population_size, time_intervals),
                    method='L-BFGS-B')

  optimized_params = result.x
  optimized_beta_s_values, optimized_beta_m_values, optimized_beta_vs_values, optimized_beta_vm_values = np.split(
      optimized_params, 4)

  _, optimized_synthetic_data = simulate_sir_model_seasonal_variation(
      population_size, initial_conditions, optimized_beta_s_values,
      optimized_beta_m_values, optimized_beta_vs_values,
      optimized_beta_vm_values, kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs,
      alpha_m, alpha_s, omega_s, omega_m, gamma_s, gamma_m, days, noise_level,
      time_intervals, should_add_noise=False)

  optimized_total_infected_data = (
      optimized_synthetic_data[:, 8] + optimized_synthetic_data[:, 9] +
      optimized_synthetic_data[:, 10] + optimized_synthetic_data[:, 11] +
      optimized_synthetic_data[:, 12] + optimized_synthetic_data[:, 13] +
      optimized_synthetic_data[:, 14] +
      optimized_synthetic_data[:, 15]) * population_size

  plt.figure(figsize=(12, 6))
  plt.plot(t_total,
           total_infected_data,
           label='Zarażeni łącznie',
           color='black')
  plt.plot(t_total,
           optimized_total_infected_data,
           label='Predykcja zarażonych',
           color='green')

  colors = ['lightgreen', 'yellow', 'orange', 'lightblue']
  season_labels = ['Wiosna', 'Lato', 'Jesień', 'Zima']
  start = 0
  for i, interval in enumerate(time_intervals):
      end = start + interval
      plt.axvspan(start, end, color=colors[i % 4], alpha=0.3, label=season_labels[i % 4])
      plt.axvline(x=start, color='k', linestyle='--')
      start = end

  plt.xlabel('Czas (dni)')
  plt.ylabel('Liczebność populacji')
  plt.title(
      'Predykcja sezonowości z optymalizacją parametrów'
  )
  plt.legend()
  plt.grid(True)
  plt.show()

  return optimized_params

def run_find_beta(population_size, initial_conditions, beta_s, beta_m, beta_vs,
                  beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m,
                  alpha_s, omega_s, omega_m, gamma_s, gamma_m, days,
                  noise_level, hospital_capacity):
    decrement_factor = 0.99
    max_iterations = 1000
    iteration = 0
    max_infected = float('inf')
    results = []
    beta_vs_values = []
    beta_vm_values = []

    while max_infected > hospital_capacity and iteration < max_iterations:
        iteration += 1
        result = simulate_sir_model_find_opt_beta(population_size, hospital_capacity,
                                                  initial_conditions, beta_s, beta_m, beta_vs,
                                                  beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm,
                                                  ni_rs, alpha_m, alpha_s, omega_s, omega_m,
                                                  gamma_s, gamma_m, days, noise_level)

        max_infected = max(result) * population_size
        results.append(result)
        beta_vs_values.append(beta_vs)
        beta_vm_values.append(beta_vm)

        if max_infected > hospital_capacity:
            beta_vs *= decrement_factor
            beta_vm *= decrement_factor

    n = len(results)
    i = n // 4
    plot_indices = [0, i, 2 * i, n - 1]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for idx, plot_idx in enumerate(plot_indices):
        t = np.linspace(0, days, days)
        axs[idx].plot(t, results[plot_idx] * population_size, 'm-', label='Zakażeni')
        axs[idx].axhline(y=hospital_capacity, color='r', linestyle='--', label='Pojemność szpitali')
        axs[idx].set_ylim(0, population_size)
        axs[idx].set_title(f"Znajdywanie optymalnego beta - Iteracja {plot_idx + 1}")
        axs[idx].set_xlabel('Czas (dni)')
        axs[idx].set_ylabel('Liczebność populacji')
        axs[idx].legend()
        axs[idx].grid(True)

        # Annotate the plot with beta values and max infected
        beta_vs_value = beta_vs_values[plot_idx]
        beta_vm_value = beta_vm_values[plot_idx]
        max_infected_value = max(results[plot_idx]) * population_size
        axs[idx].text(0.5, 0.9, f"Beta_vs: {beta_vs_value:.4f}", transform=axs[idx].transAxes, fontsize=10,
                      verticalalignment='top')
        axs[idx].text(0.5, 0.85, f"Beta_vm: {beta_vm_value:.4f}", transform=axs[idx].transAxes, fontsize=10,
                      verticalalignment='top')
        axs[idx].text(0.5, 0.8, f"Max infected: {max_infected_value:.0f}", transform=axs[idx].transAxes, fontsize=10,
                      verticalalignment='top')

    plt.tight_layout()
    plt.show()

    return beta_vs_values[-1], beta_vm_values[-1]

# New variant simulation and saving synthetic data
def run_new_variant(population_size, initial_conditions, beta_s, beta_m, beta_vs,
                    beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m,
                    alpha_s, omega_s, omega_m, gamma_s, gamma_m, days, noise_level,
                    new_variant_start, beta_s_variant, beta_m_variant,
                    beta_vs_variant, beta_vm_variant, kappa_s_variant,
                    kappa_m_variant, omega_s_variant, omega_m_variant,
                    ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant,
                    filename):
    t, synthetic_data = simulate_sir_model_new_variant(
        population_size, initial_conditions, beta_s, beta_m, beta_vs, beta_vm,
        kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
        omega_m, gamma_s, gamma_m, days, noise_level, new_variant_start,
        beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
        kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
        ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant)

    save_synthetic_data(filename, t, synthetic_data)
    plot_results_new_variant(t, synthetic_data, population_size, new_variant_start)


# Simulate SIR model with new variant
def simulate_sir_model_new_variant(population_size, initial_conditions, beta_s, beta_m, beta_vs, beta_vm,
                                   kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                                   omega_m, gamma_s, gamma_m, days, noise_level, new_variant_start,
                                   beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
                                   kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
                                   ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant, should_add_noise=True):

    t = np.linspace(0, days, days)
    result = odeint(vape_sir_model_new_variant,
                    initial_conditions,
                    t,
                    args=(beta_s, beta_m, beta_vs, beta_vm, kappa_s, kappa_m,
                          ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                          omega_m, gamma_s, gamma_m, new_variant_start,
                          beta_s_variant, beta_m_variant, beta_vs_variant,
                          beta_vm_variant, kappa_s_variant, kappa_m_variant,
                          omega_s_variant, omega_m_variant, ni_s_variant,
                          ni_m_variant, ni_rs_variant, ni_rm_variant))
    result = rectify_negative_values(result)
    if should_add_noise:
      result = add_noise(result, noise_level)

    return t, result

# Save synthetic data
def save_synthetic_data(filename, t_total, synthetic_data):
    df = pd.DataFrame(synthetic_data,
                      columns=[
                          'S1s', 'S1m', 'S2s', 'S2m', 'Sns', 'Snm', 'Sds', 'Sdm',
                          'I1s', 'I1m', 'I2s', 'I2m', 'Ins', 'Inm', 'Ids', 'Idm',
                          'Rvm', 'Rvs', 'Rns', 'Rnm', 'Rds', 'Rdm', 'Vs', 'Vm'
                      ])
    df['Time'] = t_total
    df.to_csv(filename, index=False)

# Load synthetic data
def load_synthetic_data(filename):
    df = pd.read_csv(filename)
    t_total = df['Time'].values
    synthetic_data = df.drop(columns=['Time']).values
    return t_total, synthetic_data

# Plot results for new variant
def plot_results_new_variant(t, result, population_size, new_variant_start):
    S1s, S1m, S2s, S2m, Sns, Snm, Sds, Sdm, I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm, Rvm, Rvs, Rns, Rnm, Rds, Rdm, Vs, Vm = result.T

    total_infected = (I1s + I1m + I2s + I2m + Ins + Inm + Ids + Idm) * population_size
    old_infected = (I1s + I2s + Ins + Ids) * population_size
    young_infected = (I1m + I2m + Inm + Idm) * population_size

    plt.figure(figsize=(12, 6))
    plt.plot(t, total_infected, label='Zarażeni łącznie', color='black')
    plt.plot(t, old_infected, label='Zarażeni starsi', color='red')
    plt.plot(t, young_infected, label='Zarażeni młodsi', color='blue')
    plt.axvline(x=new_variant_start, color='green', linestyle='--', label='Pojawienie się nowego wariantu')

    plt.xlabel('Czas (dni)')
    plt.ylabel('Liczebność populacji')
    plt.title(f'Model SIR - nowy wariant')
    plt.legend()
    plt.grid(True)
    plt.show()

# Objective function for optimization
def objective_function_new_variant(params, initial_conditions, t_total, observed_data,
                                   population_size, new_variant_start):
    beta_s, beta_m, beta_vs, beta_vm, beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant = params

    _, result_total = simulate_sir_model_new_variant(
        population_size, initial_conditions, beta_s, beta_m, beta_vs, beta_vm,
        kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
        omega_m, gamma_s, gamma_m, len(t_total), noise_level, new_variant_start,
        beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
        kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
        ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant, should_add_noise=False)

    I1s, I1m, I2s, I2m, Ins, Inm, Ids, Idm = result_total[:, 8:16].T
    predicted_infected = (I1s + I1m + I2s + I2m + Ins + Inm + Ids + Idm) * population_size

    error = observed_data - predicted_infected
    return np.sum(error**2)

# Run optimization for new variant

def run_optimization_new_variant(population_size, initial_conditions, new_variant_start,
                                 worse_initial_guess, filename):
    try:
        t_total, synthetic_data = load_synthetic_data(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Running simulation to create the file.")
        run_new_variant(
            population_size, initial_conditions, beta_s, beta_m, beta_vs, beta_vm,
            kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
            omega_m, gamma_s, gamma_m, days, noise_level, new_variant_start,
            beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
            kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
            ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant, filename)
        t_total, synthetic_data = load_synthetic_data(filename)

    total_infected_data = (
        synthetic_data[:, 8] + synthetic_data[:, 9] + synthetic_data[:, 10] +
        synthetic_data[:, 11] + synthetic_data[:, 12] + synthetic_data[:, 13] +
        synthetic_data[:, 14] + synthetic_data[:, 15]) * population_size

    result = minimize(objective_function_new_variant,
                      worse_initial_guess,
                      args=(initial_conditions, t_total, total_infected_data,
                            population_size, new_variant_start),
                      method='L-BFGS-B')

    optimized_params = result.x
    beta_s, beta_m, beta_vs, beta_vm, beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant = optimized_params

    t, optimized_synthetic_data = simulate_sir_model_new_variant(
        population_size, initial_conditions, beta_s, beta_m, beta_vs, beta_vm,
        kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
        omega_m, gamma_s, gamma_m, len(t_total), noise_level, new_variant_start,
        beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
        kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
        ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant, should_add_noise=False)

    optimized_total_infected_data = (
        optimized_synthetic_data[:, 8] + optimized_synthetic_data[:, 9] +
        optimized_synthetic_data[:, 10] + optimized_synthetic_data[:, 11] +
        optimized_synthetic_data[:, 12] + optimized_synthetic_data[:, 13] +
        optimized_synthetic_data[:, 14] +
        optimized_synthetic_data[:, 15]) * population_size

    plt.figure(figsize=(12, 6))
    plt.plot(t_total,
             total_infected_data,
             label='Zarażeni (Dane)',
             color='black')
    plt.plot(t_total,
             optimized_total_infected_data,
             label='Predykcja zarażonych (Optymalizacja)',
             color='green')
    plt.axvline(x=new_variant_start, color='green', linestyle='--', label='Pojawienie się nowego wariantu')
    plt.xlabel('Czas (dni)')
    plt.ylabel('Liczebność populacji')
    plt.title('Predykcja z optymalizacją parametrów - Nowy wariant')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimized_params


# Run simulation, save synthetic data, and run optimization for new variant
def run_new_variant_simulation_and_prediction(population_size, initial_conditions, beta_s, beta_m, beta_vs, beta_vm,
                                              kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
                                              omega_m, gamma_s, gamma_m, days, noise_level, new_variant_start,
                                              beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
                                              kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
                                              ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant, filename,
                                              worse_initial_guess):
    t, synthetic_data = simulate_sir_model_new_variant(
        population_size, initial_conditions, beta_s, beta_m, beta_vs, beta_vm,
        kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s, omega_s,
        omega_m, gamma_s, gamma_m, days, noise_level, new_variant_start,
        beta_s_variant, beta_m_variant, beta_vs_variant, beta_vm_variant,
        kappa_s_variant, kappa_m_variant, omega_s_variant, omega_m_variant,
        ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant)

    save_synthetic_data(filename, t, synthetic_data)
    plot_results_new_variant(t, synthetic_data, population_size, new_variant_start)

    optimized_params = run_optimization_new_variant(
        population_size, initial_conditions, new_variant_start, worse_initial_guess, filename)

    print("Optimized parameters:", optimized_params)

# Unified parameters
beta_s = 0.06
beta_m = 0.02
beta_vs = 0.45
beta_vm = 0.35
kappa_s = 1 / 140
kappa_m = 1 / 210
ni_m = 0.81
ni_s = 0.95
ni_rm = 0.66
ni_rs = 0.83
alpha_m = 0.75
alpha_s = 0.85
omega_s = 1 / 280
omega_m = 1 / 210
gamma_s = 0.067
gamma_m = 0.071
beta_s_variant = 0.6
beta_m_variant = 0.2
beta_vs_variant = 1
beta_vm_variant = 1
kappa_s_variant = 1 / 140
kappa_m_variant = 1 / 210
omega_s_variant = 1 / 2 * 1 / 280
omega_m_variant = 1 / 2 * 1 / 210
ni_m_variant = 0.81
ni_s_variant = 0.95
ni_rm_variant = 0.66
ni_rs_variant = 0.83
alpha_m_variant = 0.01875
alpha_s_variant = 0.02125

noise_level = 0.0005

years = 2
days = 360 * years

# Example usage:
population_size = 100000
initial_conditions = [0.0625] * 16 + [0] * 8
initial_conditions_hosp = [0.12] * 8 + [0.005] * 8 + [0] * 8
time_intervals = [90, 90, 90, 90] * years

worse_variant_initial_guess = np.array([0.02, 0.02, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5])
better_variant_initial_guess = np.array([0.05, 0.02, 0.4, 0.3, 0.5, 0.15, 0.8, 0.8])

worse_seasonal_initial_guess = np.concatenate([
    np.array([0.1, 0.3, 0.1, 0.1] * years),
    np.array([0.1, 0.3, 0.1, 0.1] * years),
    np.array([0.3, 0.7, 0.3, 0.3] * years),
    np.array([0.3, 0.7, 0.3, 0.3] * years)
])

better_seasonal_initial_guess = np.concatenate([
    np.array([0.05, 0.4, 0.05, 0.05] * years),
    np.array([0.01, 0.7, 0.05, 0.1] * years),
    np.array([0.4, 0.6, 0.25, 0.45] * years),
    np.array([0.3, 0.8, 0.25, 0.55] * years)
])

initial_seasonal_guess = better_seasonal_initial_guess

initial_variant_guess = better_variant_initial_guess

# Seasonal beta values (higher in summer, lower in winter)
beta_s_values = [0.06, 0.45, 0.03, 0.06] * years
beta_m_values = [0.02, 0.75, 0.02, 0.02] * years
beta_vs_values = [0.45, 0.7, 0.3, 0.45] * years
beta_vm_values = [0.35, 0.9, 0.3, 0.35] * years

hospital_capacity = 0.1 * population_size
new_variant_start = int(days/2)  # Middle of the simulation

# Generate and save synthetic data
filename_seasonal = 'synthetic_data_seasonal.csv'
filename_variant = 'synthetic_data_variant.csv'

# Run normal simulation
run_simulation(population_size, initial_conditions, beta_s, beta_m, beta_vs,
               beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m,
               alpha_s, omega_s, omega_m, gamma_s, gamma_m, days, noise_level,
               'Normal Simulation')

# Hospital capacity
run_find_beta(population_size, initial_conditions_hosp, beta_s, beta_m,
              beta_vs, beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs,
              alpha_m, alpha_s, omega_s, omega_m, gamma_s, gamma_m, days,
              noise_level, hospital_capacity)

# Run seasonal simulation and save data
run_seasonal_simulation(population_size, initial_conditions, beta_s_values,
                        beta_m_values, beta_vs_values, beta_vm_values, kappa_s,
                        kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m, alpha_s,
                        omega_s, omega_m, gamma_s, gamma_m, days, noise_level,
                        time_intervals, filename_seasonal)


# New variant simulation and saving synthetic data
run_new_variant(population_size, initial_conditions, beta_s, beta_m, beta_vs,
                beta_vm, kappa_s, kappa_m, ni_m, ni_s, ni_rm, ni_rs, alpha_m,
                alpha_s, omega_s, omega_m, gamma_s, gamma_m, days, noise_level,
                new_variant_start, beta_s_variant, beta_m_variant,
                beta_vs_variant, beta_vm_variant, kappa_s_variant,
                kappa_m_variant, omega_s_variant, omega_m_variant,
                ni_s_variant, ni_m_variant, ni_rs_variant, ni_rm_variant, filename_variant)

# Run optimization using the saved synthetic data
run_optimization_seasonal(population_size, initial_conditions, time_intervals,
                 initial_seasonal_guess, filename_seasonal)

# Run optimization using the saved synthetic data for new variant
run_optimization_new_variant(population_size, initial_conditions, new_variant_start,
                             initial_variant_guess, filename_variant)
