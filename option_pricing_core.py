
import numpy as np
from scipy.stats import norm
import math
# Import additional math functions for new models
from math import exp, log, sqrt, factorial 
import pandas as pd
import warnings

# ==============================================================================
# 0. CONFIGURATION AND THRESHOLDS
# ==============================================================================
class HybridAnalysisConfig:
    """Hybrid configuration parameters for analysis thresholds"""
    
    # M6 FIX: Renamed valuation thresholds for clarity (Model Price - Market Price)
    # If Mispricing > 0.50, Model > Market -> Market is UNDERVALUED (Potential Buy)
    MISPRICING_BUY_THRESHOLD = 0.50
    # If Mispricing < -0.50, Model < Market -> Market is OVERVALUED (Potential Sell)
    MISPRICING_SELL_THRESHOLD = -0.50 
    
    # PCR sentiment thresholds
    PCR_BULLISH = 1.10
    PCR_BEARISH = 0.90
    
    # Greeks thresholds (Hybrid and Absolute)
    # m12 FIX: Lowered for better sensitivity in delta risk
    GAMMA_HIGH_ABSOLUTE = 0.001 
    VEGA_HIGH_PCT = 1.5           # 1.5% of underlying (Used for relative Vega Risk)
    VEGA_HIGH_ABSOLUTE = 1500     # Absolute fallback for Vega
    RHO_SIGNIFICANT_PCT = 0.1     # 0.1% of underlying (Used for relative Rho Risk)
    RHO_SIGNIFICANT_ABSOLUTE = 100 # Absolute fallback for Rho (per 1% rate change)
    THETA_SIGNIFICANT_PCT = 0.05  # 5% of option price per day
    
    # IV comparison threshold
    IV_DIFFERENCE_PCT = 0.03 # 3% difference in IV to flag skew

    # Moneyness thresholds
    DELTA_DEEP_ITM = 0.75
    DELTA_ATM_LOWER = 0.30

    # [INSERT THIS NEW METHOD INSIDE HybridAnalysisConfig CLASS]
    @staticmethod
    def get_delta_implication(delta, option_type='call'):
        """Determine directional risk based on Delta."""
        if np.isnan(delta): return "N/A"
        
        abs_delta = abs(delta)
        direction = "Long" if delta > 0 else "Short"
        
        # Determine sensitivity description
        if abs_delta > 0.75:
             sensitivity = "High directional sensitivity (Acts like the underlying)."
        elif abs_delta < 0.30:
             sensitivity = "Low directional sensitivity (Far OTM)."
        else:
             sensitivity = "Moderate directional sensitivity."

        return (f"{sensitivity}\n"
                f"  → Delta: {delta:.2f} ({abs_delta*100:.1f}% probability of expiring ITM)\n"
                f"  → Equivalent to holding {abs_delta*100:.0f} shares of the underlying ({direction}).")

    @staticmethod
    def get_gamma_implication(gamma):
        """Determine delta risk based on Gamma absolute threshold."""
        if np.isnan(gamma): return "N/A"
        if gamma > HybridAnalysisConfig.GAMMA_HIGH_ABSOLUTE:
            return "Highly Responsive/Volatile: Delta changes rapidly with underlying moves (high Gamma)."
        else:
            return "Stable/Less Responsive: Delta changes slowly with underlying moves (low Gamma)."
            
    @staticmethod
    def get_vega_implication(vega, underlying_price):
        """Hybrid Vega analysis with both metrics"""
        if np.isnan(vega): return "N/A"
        # Ensure underlying_price is not zero to avoid division error
        if underlying_price == 0: return "Vega analysis unavailable (zero underlying price)."
        vega_pct = (abs(vega) / underlying_price) * 100
        is_high_pct = vega_pct > HybridAnalysisConfig.VEGA_HIGH_PCT
        is_high_abs = abs(vega) > HybridAnalysisConfig.VEGA_HIGH_ABSOLUTE
        
        if is_high_pct or is_high_abs:
            return (f"Volatility Risk is High.\n"
                    f"  → Vega: {vega:.2f} ({vega_pct:.3f}% of underlying)\n"
                    f"  → 1% IV change ≈ ₹{abs(vega)/100:.2f} price change")
        else:
            return (f"Volatility Risk is Moderate.\n"
                    f"  → Vega: {vega:.2f} ({vega_pct:.3f}% of underlying)\n"
                    f"  → 1% IV change ≈ ₹{abs(vega)/100:.2f} price change")

    @staticmethod
    def get_rho_implication(rho, underlying_price, option_type='call', asset_type='Stock'):
        """Hybrid Rho analysis (Rho should be the annual value here)"""
        if np.isnan(rho): return "N/A"
        rho_per_1pct = rho / 100 
        # Ensure underlying_price is not zero
        if underlying_price == 0: return "Rho analysis unavailable (zero underlying price)."
        rho_pct = (abs(rho_per_1pct) / underlying_price) * 100 # % of underlying
        
        is_significant = (rho_pct > HybridAnalysisConfig.RHO_SIGNIFICANT_PCT or 
                          abs(rho_per_1pct) > HybridAnalysisConfig.RHO_SIGNIFICANT_ABSOLUTE)
        
        direction = "increases" if rho > 0 else "decreases"
        
        # m10 FIX: Add explanation for commodity Rho
        commodity_note = ""
        if asset_type == 'Commodity':
            commodity_note = ("For futures options (commodities), Rho is typically small and often "
                              "negative for both calls and puts, as changes in 'r' primarily affect "
                              "discounting, not the underlying futures price's drift (since q=r).")
        
        rho_details = (f"  → Rho: {rho_per_1pct:+.2f} per 1% rate change ({rho_pct:.3f}% of underlying)\n"
                       f"  → If rates rise 1%, option value {direction} by ₹{abs(rho_per_1pct):.2f}")
        
        if is_significant:
            return (f"Significant rate risk for {option_type}.\n" + 
                    rho_details + 
                    (f"\n  NOTE: {commodity_note}" if commodity_note else ""))
        else:
            return (f"Moderate rate risk for {option_type}.\n" + 
                    rho_details + 
                    (f"\n  NOTE: {commodity_note}" if commodity_note else ""))


    @staticmethod
    def get_theta_implication(theta, model_price):
        """Determine if Theta (time decay) is significant based on % of option price.
        Theta input must be the daily value."""
        if np.isnan(theta) or np.isnan(model_price): return "N/A"
        if model_price <= 0.01:
            return f"Option near worthless. Theta: {theta:.4f} (decay rate undefined)."
        
        try:
            theta_pct = abs(theta) / model_price
        except ZeroDivisionError:
            return f"Moderate time decay.\n  → Daily Theta: {theta:.4f} (decay rate undefined)."
        
        if theta_pct > HybridAnalysisConfig.THETA_SIGNIFICANT_PCT:
            return (f"Significant time decay.\n"
                    f"  → Daily Theta: {theta:.4f} ({theta_pct:.2%} of option value)\n"
                    f"  → Losing ₹{abs(theta):.2f} per day. Favorable for sellers.")
        else:
            return (f"Moderate time decay.\n"
                    f"  → Daily Theta: {theta:.4f} ({theta_pct:.2%} of option value)\n"
                    f"  → Losing ₹{abs(theta):.2f} per day.")


# ==============================================================================
# 1. INPUT VALIDATION
# ==============================================================================

# In option_pricing_core.py

def validate_inputs(K, T_days, r, sigma, S=None, F=None, q=None, asset_type='Unknown',
                    lam=0, mu_j=0, sigma_j=0, N_bin=0, M_mc=0): # M7 FIX: Added extended model params
    """Validate all inputs before processing"""
    errors = []

    # --- Check for None (empty fields) first ---
    if K is None: errors.append("Strike (K) cannot be empty.")
    if T_days is None: errors.append("Days to Expiry cannot be empty.")
    if r is None: errors.append("Risk-Free Rate (r) cannot be empty.")
    if sigma is None: errors.append("User Volatility (σ) cannot be empty.")
    if lam is None: errors.append("λ (Jump) cannot be empty.")
    if mu_j is None: errors.append("μ_J cannot be empty.")
    if sigma_j is None: errors.append("σ_J cannot be empty.")
    if N_bin is None: errors.append("Binomial Steps (N) cannot be empty.")
    if M_mc is None: errors.append("MC Paths (M) cannot be empty.")
    
    # Check for S or F based on asset type
    if asset_type in ['Stock', 'Index'] and S is None:
        errors.append("Spot (S) cannot be empty for Stock/Index.")
    elif asset_type == 'Commodity' and F is None:
        errors.append("Futures (F) cannot be empty for Commodity.")

    # If any required fields are None, return early as further checks will fail
    if errors:
        raise ValueError("\n".join(errors))

    # --- Now we know they are numbers, check their values ---
    T = T_days / 365.0
    
    if K <= 0: errors.append("Strike price must be positive.")
    if T <= 0: errors.append("Time to expiration must be positive.")
    if T > 1.0: errors.append("Time to expiration (in years) is highly unusual (> 1 year). Please check.")
    if sigma < 0: errors.append("Volatility cannot be negative.")
    if sigma > 1.0: errors.append("Volatility is unusually high (> 100%). Please check.")
    if r < -0.10 or r > 0.50: errors.append("Risk-Free rate is outside the typical range of [-10%, 50%]. Please check.")
    
    if S is not None and S <= 0: errors.append("Spot price must be positive.")
    if F is not None and F <= 0: errors.append("Futures price must be positive.")
    if q is not None and (q < 0 or q > 0.10): errors.append("Dividend yield is outside the typical range of [0%, 10%]. Please check.")
        
    underlying = S if S is not None else F
    if underlying is not None and (underlying / K > 10 or K / underlying > 10):
          errors.append("Underlying price and Strike price are too far apart (deep OTM/ITM). Please check inputs.")

    # M7 FIX: Extended model parameter validation
    if lam < 0: errors.append("Jump Intensity (lambda) cannot be negative.")
    if sigma_j < 0: errors.append("Std Dev Log Jump Size (sigma_J) cannot be negative.")
    if abs(mu_j) > 1.0: errors.append("Mean Log Jump Size (mu_J) is highly unusual (magnitude > 1.0). Please check.")
    if N_bin < 0: errors.append("Binomial Steps (N) cannot be negative.")
    if M_mc < 0: errors.append("Monte Carlo Paths (M) cannot be negative.")

    if errors:
        raise ValueError("\n".join(errors))
        
    return True


# ==============================================================================
# 2. CORE PRICING AND GREEKS FUNCTIONS (BSM for Stock/Index)
# ==============================================================================

def bsm_price_and_vega_only(S, K, T, r, sigma, q):
    """Calculates Call/Put price and Vega using BSM model (for efficient IV iteration)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma <= 1e-6:
            call_price = np.maximum(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            put_price = np.maximum(0, K * np.exp(-r * T) - S * np.exp(-q * T))
            return call_price, put_price, 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    return call_price, put_price, vega


def black_scholes_merton(S, K, T, r, sigma, q):
    """Calculates BSM price and the core d1/d2 for Greeks."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma <= 1e-6:
            call_price = np.maximum(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            put_price = np.maximum(0, K * np.exp(-r * T) - S * np.exp(-q * T))
            return call_price, put_price, 0.0, 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return call_price, put_price, d1, d2


def calculate_greeks_BSM(S, K, d1, d2, T, r, q, sigma):
    """Calculates all Greeks for the BSM model (Annualized values)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma <= 1e-6:
            # Return NaNs instead of 0s if inputs are invalid for Greeks
            return (np.nan,)*8
            
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_prime_d1 = norm.pdf(d1)
        
        # Core Greeks shared by Call and Put (Vega, Gamma)
        vega = S * np.exp(-q * T) * N_prime_d1 * np.sqrt(T)
        gamma = N_prime_d1 * np.exp(-q * T) / (S * sigma * np.sqrt(T))
        
        # Delta
        delta_call = N_d1 * np.exp(-q * T)
        delta_put = (N_d1 - 1) * np.exp(-q * T)
        
        # Theta (Annualized)
        theta_base = -(S * np.exp(-q * T) * N_prime_d1 * sigma) / (2 * np.sqrt(T))
        theta_call = theta_base - r * K * np.exp(-r * T) * N_d2 + q * S * np.exp(-q * T) * N_d1
        theta_put = theta_base + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
        
        # Rho (Annualized)
        rho_call = K * T * np.exp(-r * T) * N_d2
        rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


# ==============================================================================
# 2.1 EXTENDED PRICING MODELS 
# ==============================================================================

# Helper BSM Call/Put Price functions used by Merton
def _bsm_price_only(S, K, T, r_adj, sigma_adj, q, r_discount, option_type='call'):
    """
    Internal helper: Calculates European Price using BSM with Dividend Yield (q).
    Uses (r_adj) as the adjusted drift rate, so (r-q) is NOT applied inside the BSM formula.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma_adj <= 1e-6:
            if option_type == 'call':
                return np.maximum(0, S * np.exp(-q * T) - K * np.exp(-r_adj * T))
            else:
                return np.maximum(0, K * np.exp(-r_adj * T) - S * np.exp(-q * T))

        d1 = (np.log(S/K) + (r_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r_discount * T) * norm.cdf(d2)
        else: # put
            price = K * np.exp(-r_discount * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            
        return price

# Binomial Tree Model (Handles American Options)
def binomial_pricing(S0, K, T, r, sigma, N, option_type='call', is_american=True, q=0):
    """Calculates American/European Option Price using the CRR Binomial Model with dividend yield q."""
    
    # Validate inputs to prevent math errors
    if T <= 0 or N <= 0:
        if option_type == 'call':
            return np.maximum(0, S0 * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            return np.maximum(0, K * np.exp(-r * T) - S0 * np.exp(-q * T))
            
    dt = T / N
    disc = np.exp(-r * dt)
    
    # CRR Model factors (u, d) and risk-neutral probability (p) with dividend correction (r-q)
    try:
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
    except OverflowError:
        # Handle cases where sigma * sqrt(dt) is too large
        if option_type == 'call':
            return np.maximum(0, S0 * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            return np.maximum(0, K * np.exp(-r * T) - S0 * np.exp(-q * T))

    # Check for arbitrage-free condition
    if not (d < np.exp((r-q) * dt) < u):
        pass # Allow calculation to proceed, may be inexact
        
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # Ensure probability is valid
    if p < 0 or p > 1:
        # This can happen if N is too small or parameters are extreme
        # Fallback to intrinsic value
        if option_type == 'call':
            return np.maximum(0, S0 * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            return np.maximum(0, K * np.exp(-r * T) - S0 * np.exp(-q * T))

    # Initialize Option Values (Final Payoffs at T)
    S_T = S0 * (u**np.arange(N+1)) * (d**np.arange(N, -1, -1))

    if option_type == 'call':
        V = np.maximum(S_T - K, 0)
    else:
        V = np.maximum(K - S_T, 0)

    # Backward Induction Loop
    for i in range(N - 1, -1, -1):
        # Value if Held (Continuation Value)
        V_next_step = V
        V = disc * (p * V_next_step[1:i+2] + (1 - p) * V_next_step[0:i+1])

        if is_american:
            # Recalculate stock prices at the current step i
            S_i = S0 * (u**np.arange(i + 1)) * (d**np.arange(i, -1, -1))

            # Value if Exercised (Intrinsic Value)
            if option_type == 'call':
                V_exercised = np.maximum(S_i - K, 0)
            else:
                V_exercised = np.maximum(K - S_i, 0)

            # Option Value is the MAXIMUM of held or exercised
            V = np.maximum(V, V_exercised)
            
    return V[0]

# Merton Jump-Diffusion Model (Handles Jump Risk)
def merton_jump_price(S, K, T, r, sigma, q, lam, mu_j, sigma_j, N_jumps=40, option_type='call'):
    """
    Calculates the Merton Jump-Diffusion European Call or Put Price.
    lam: Jump Intensity (lambda)
    mu_j: Mean Log Jump Size (mu_J)
    sigma_j: Std Dev Log Jump Size (sigma_J)
    """
    
    merton_price = 0.0
    
    # Calculate the expected jump multiplier (m = E[e^J])
    m = np.exp(mu_j + 0.5 * sigma_j**2)
    
    # Risk-Neutral Jump Intensity (lambda') = lam * m
    lam_prime = lam * m
    
    for k in range(N_jumps):
        # 1. Calculate the Poisson Probability Weight: P(k jumps)
        try:
            poisson_prob = (np.exp(-lam_prime * T) * (lam_prime * T)**k) / factorial(k)
        except (OverflowError, ValueError):
            break 
            
        # 2. Adjusted Drift Rate (r_k)
        r_q = r - q # Base dividend-adjusted drift
        r_k = r_q - lam * (m - 1) + (k * (mu_j + 0.5*sigma_j**2)) / T
        
        # 3. Adjusted Volatility (sigma_k)
        sigma_k = np.sqrt(sigma**2 + (k * sigma_j**2) / T)
        
        # 4. Weighted BSM Sum
        conditional_bsm_price = _bsm_price_only(S, K, T, r_k, sigma_k, q, r, option_type)
        
        merton_price += poisson_prob * conditional_bsm_price
        
    return merton_price


# Monte Carlo Simulation (Flexible benchmark for European Options)
def monte_carlo_price(S0, K, T, r, sigma, q, M_paths=100000, option_type='call'):
    """Estimates the European Option Price using Monte Carlo Simulation (Terminal Price)."""
    
    if T <= 0:
        if option_type == 'call':
            return np.maximum(0, S0 * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            return np.maximum(0, K * np.exp(-r * T) - S0 * np.exp(-q * T))
            
    # Pre-calculate the components of the exponent for efficiency
    drift = (r - q - 0.5 * sigma**2) * T
    vol_term = sigma * np.sqrt(T)
    
    # 1. Generate M_paths random draws (Z)
    Z = np.random.normal(size=M_paths)
    
    # 2. Calculate M_paths final stock prices (ST) using the vectorized GBM formula
    ST = S0 * np.exp(drift + vol_term * Z)
    
    # 3. Calculate payoffs for each path
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else: # 'put'
        payoffs = np.maximum(K - ST, 0)
        
    # 4. Discount the average payoff to present value
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price

# ==============================================================================
# 2.2 EXTENDED GREEKS FUNCTIONS (MERTON & NUMERICAL)
# ==============================================================================

def _bsm_greeks_internal(S, K, T, r_adj, sigma_adj, q, r_discount):
    """Internal helper: Calculates all BSM Greeks for a single Merton component."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma_adj <= 1e-6:
            return (np.nan,)*8
            
        d1 = (np.log(S/K) + (r_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_prime_d1 = norm.pdf(d1)

        # Core Greeks
        vega = S * np.exp(-q * T) * N_prime_d1 * np.sqrt(T)
        gamma = N_prime_d1 * np.exp(-q * T) / (S * sigma_adj * np.sqrt(T))

        # Delta
        delta_call = N_d1 * np.exp(-q * T)
        delta_put = (N_d1 - 1) * np.exp(-q * T)

        # Theta (Annualized)
        theta_base = -(S * np.exp(-q * T) * N_prime_d1 * sigma_adj) / (2 * np.sqrt(T))
        # Note: r_adj is the drift, r is the discount rate (from BSM formula)
        # We must use r_adj in the discounting terms of the BSM Theta formula
        theta_call = theta_base - r_discount * K * np.exp(-r_discount * T) * N_d2 + q * S * np.exp(-q * T) * N_d1
        theta_put = theta_base + r_discount * K * np.exp(-r_discount * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)

        # Rho (Annualized)
        # This is dC/dr_adj
        rho_call = K * T * np.exp(-r_discount * T) * N_d2
        rho_put = -K * T * np.exp(-r_discount * T) * norm.cdf(-d2)

    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


def calculate_greeks_merton(F_or_S, K, T, r, sigma, q, lam, mu_j, sigma_j, asset_type, N_jumps=40):
    """Calculates all Greeks for the Merton Jump-Diffusion model."""
    
    # Initialize Greeks
    delta_call_m, delta_put_m = 0.0, 0.0
    gamma_m = 0.0
    vega_m = 0.0
    theta_call_m, theta_put_m = 0.0, 0.0
    # Rho is calculated numerically outside the loop

    # Constants
    m = np.exp(mu_j + 0.5 * sigma_j**2)
    lam_prime = lam * m
    log_m = mu_j + 0.5 * sigma_j**2

    for k in range(N_jumps):
        # 1. Poisson Probability (pi_k)
        try:
            poisson_prob = (np.exp(-lam_prime * T) * (lam_prime * T)**k) / factorial(k)
        except (OverflowError, ValueError):
            break # k is too large
            
        # 2. Adjusted Parameters (r_k, sigma_k)
        r_q = r - q
        r_k = r_q - lam * (m - 1) + (k * log_m) / T
        sigma_k = np.sqrt(sigma**2 + (k * sigma_j**2) / T)

        # 3. Get BSM component Greeks
        (d_c, d_p, g, v_k, t_c, t_p, r_c_k, r_p_k) = _bsm_greeks_internal(F_or_S, K, T, r_k, sigma_k, q, r)
        
        if np.isnan(d_c): continue # Skip if BSM greeks failed

        # 4. Calculate Weighted Sums (applying Chain Rule where needed)
        
        # Delta & Gamma (Simple weighted sum)
        delta_call_m += poisson_prob * d_c
        delta_put_m += poisson_prob * d_p
        gamma_m += poisson_prob * g
        
        # Vega (Chain Rule: d(sigma_k)/d(sigma) = sigma / sigma_k)
        if sigma_k > 1e-9: # Avoid division by zero
            vega_m += poisson_prob * v_k * (sigma / sigma_k)

        # Theta (Full Chain Rule)
        d_pi_k_dT = (k/T - lam_prime) * poisson_prob if T > 1e-9 else 0
        d_r_k_dT = -k * log_m / (T**2) if T > 1e-9 else 0
        d_sigma_k_dT = -k * (sigma_j**2) / (2 * T**2 * sigma_k) if T > 1e-9 and sigma_k > 1e-9 else 0

        price_c_k = _bsm_price_only(F_or_S, K, T, r_k, sigma_k, q, r, 'call')
        price_p_k = _bsm_price_only(F_or_S, K, T, r_k, sigma_k, q, r, 'put')
        
        theta_call_m += d_pi_k_dT * price_c_k + poisson_prob * (t_c + r_c_k * d_r_k_dT + v_k * d_sigma_k_dT)
        theta_put_m += d_pi_k_dT * price_p_k + poisson_prob * (t_p + r_p_k * d_r_k_dT + v_k * d_sigma_k_dT)

    # 5. RHO FIX: Calculate Rho numerically
    epsilon_r_safe = 0.0001 # 1 basis point
    r_up = r + epsilon_r_safe
    r_down = r - epsilon_r_safe

    # *** C2 FIX (Merton): Apply the correct q=r bump for Commodities ***
    q_up = q
    q_down = q
    if asset_type == 'Commodity':
        q_up = r_up
        q_down = r_down
    else:
        q_up = q
        q_down = q
    
    price_r_up_call = merton_jump_price(F_or_S, K, T, r_up, sigma, q_up, lam, mu_j, sigma_j, N_jumps, 'call')
    price_r_down_call = merton_jump_price(F_or_S, K, T, r_down, sigma, q_down, lam, mu_j, sigma_j, N_jumps, 'call')
    rho_call_m = (price_r_up_call - price_r_down_call) / (2 * epsilon_r_safe)

    price_r_up_put = merton_jump_price(F_or_S, K, T, r_up, sigma, q_up, lam, mu_j, sigma_j, N_jumps, 'put')
    price_r_down_put = merton_jump_price(F_or_S, K, T, r_down, sigma, q_down, lam, mu_j, sigma_j, N_jumps, 'put')
    rho_put_m = (price_r_up_put - price_r_down_put) / (2 * epsilon_r_safe)


    return delta_call_m, delta_put_m, gamma_m, vega_m, theta_call_m, theta_put_m, rho_call_m, rho_put_m


def calculate_greeks_numerical(pricing_func, F_or_S, K, T, r, sigma, N, q, is_american, asset_type, bump_frac=0.001):
    """Calculates all Greeks for a numerical model (like Binomial) using 'bump and revalue'."""
    
    # Define bumps based on fractional value
    epsilon_S = F_or_S * bump_frac
    epsilon_sigma = sigma * bump_frac
    
    # Handle zero or small values
    if epsilon_S == 0: epsilon_S = 0.001
    if epsilon_sigma == 0: epsilon_sigma = 0.0001
    epsilon_r_safe = 0.0001 # Use fixed bump for r

    # 1. Delta & Gamma
    price_S_up = pricing_func(F_or_S + epsilon_S, K, T, r, sigma, N, 'call', is_american, q)
    price_S_down = pricing_func(F_or_S - epsilon_S, K, T, r, sigma, N, 'call', is_american, q)
    price_S_mid = pricing_func(F_or_S, K, T, r, sigma, N, 'call', is_american, q)
    
    delta_call = (price_S_up - price_S_down) / (2 * epsilon_S)
    gamma = (price_S_up - 2 * price_S_mid + price_S_down) / (epsilon_S**2)
    
    price_S_up_put = pricing_func(F_or_S + epsilon_S, K, T, r, sigma, N, 'put', is_american, q)
    price_S_down_put = pricing_func(F_or_S - epsilon_S, K, T, r, sigma, N, 'put', is_american, q)
    delta_put = (price_S_up_put - price_S_down_put) / (2 * epsilon_S)
    
    # 2. Vega
    price_v_up = pricing_func(F_or_S, K, T, r, sigma + epsilon_sigma, N, 'call', is_american, q)
    price_v_down = pricing_func(F_or_S, K, T, r, sigma - epsilon_sigma, N, 'call', is_american, q)
    vega = (price_v_up - price_v_down) / (2 * epsilon_sigma)
    
    # 3. Theta (T is time to expiry, so we check T-epsilon)
    # Use a smaller bump for T to avoid negative time
    epsilon_T_safe = min(T * 0.01, 1/365.0) # 1% of T or 1 day, whichever is smaller
    if epsilon_T_safe <= 1e-9:
        theta_call, theta_put = 0.0, 0.0
    else:
        # Note: Time to expiry is T. One day of decay means T becomes T - 1/365. 
        # To calculate annual theta, we use the central difference formula:
        price_T_up = pricing_func(F_or_S, K, T + epsilon_T_safe, r, sigma, N, 'call', is_american, q)
        price_T_down = pricing_func(F_or_S, K, T - epsilon_T_safe, r, sigma, N, 'call', is_american, q)
        theta_call = -(price_T_up - price_T_down) / (2 * epsilon_T_safe) # Negative sign for time decay
        
        price_T_up_put = pricing_func(F_or_S, K, T + epsilon_T_safe, r, sigma, N, 'put', is_american, q)
        price_T_down_put = pricing_func(F_or_S, K, T - epsilon_T_safe, r, sigma, N, 'put', is_american, q)
        theta_put = -(price_T_up_put - price_T_down_put) / (2 * epsilon_T_safe)

    # 4. Rho
    epsilon_r_safe = 0.0001
    r_up = r + epsilon_r_safe
    r_down = r - epsilon_r_safe

    # *** C2 Fix (Binomial): Maintain F=Se^(r-q)T assumption for Commodity ***
    q_up = q
    q_down = q
    if asset_type == 'Commodity':
        # For Binomial (American), r and q must be bumped together to maintain F=S 
        # which is the futures pricing model assumption underlying Black-76
        q_up = r_up
        q_down = r_down
    else:
        q_up = q
        q_down = q
    # *** END FIX ***

    price_r_up = pricing_func(F_or_S, K, T, r_up, sigma, N, 'call', is_american, q_up)
    price_r_down = pricing_func(F_or_S, K, T, r_down, sigma, N, 'call', is_american, q_down)
    rho_call = (price_r_up - price_r_down) / (2 * epsilon_r_safe)
    
    price_r_up_put = pricing_func(F_or_S, K, T, r_up, sigma, N, 'put', is_american, q_up)
    price_r_down_put = pricing_func(F_or_S, K, T, r_down, sigma, N, 'put', is_american, q_down)
    rho_put = (price_r_up_put - price_r_down_put) / (2 * epsilon_r_safe)

    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


# ==============================================================================
# 3. CORE PRICING AND GREEKS FUNCTIONS (Black-76 for Commodity)
# ==============================================================================

# C3 FIX: Restoring Black-76 functions as requested
def black76_price_and_vega_only(F, K, T, r, sigma):
    """Calculates Call/Put price and Vega using Black-76 model (for efficient IV iteration)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma <= 1e-6:
            call_price = np.maximum(0, F - K) * np.exp(-r * T)
            put_price = np.maximum(0, K - F) * np.exp(-r * T)
            return call_price, put_price, 0.0

        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        vega = F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
        d2 = d1 - sigma * np.sqrt(T)
        discount_factor = np.exp(-r * T)

        call_price = discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
        put_price = discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    return call_price, put_price, vega


def black_76(F, K, T, r, sigma):
    """Calculates Black-76 price and the core d1/d2 for Greeks."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma <= 1e-6:
            call_price = np.maximum(0, F - K) * np.exp(-r * T)
            put_price = np.maximum(0, K - F) * np.exp(-r * T)
            return call_price, put_price, 0.0, 0.0

        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        discount_factor = np.exp(-r * T)

        call_price = discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
        put_price = discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    return call_price, put_price, d1, d2


def calculate_greeks_Black76(F, K, d1, d2, T, r, sigma):
    """Calculates all Greeks for the Black-76 model (Annualized values)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if T <= 0 or sigma <= 1e-6:
            return (np.nan,)*8
            
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_prime_d1 = norm.pdf(d1)
        discount_factor = np.exp(-r * T)
        
        # Core Greeks shared by Call and Put (Vega, Gamma)
        vega = F * discount_factor * N_prime_d1 * np.sqrt(T)
        gamma = N_prime_d1 * discount_factor / (F * sigma * np.sqrt(T))
        
        # Delta (Forward Delta)
        delta_call = discount_factor * N_d1
        delta_put = discount_factor * (N_d1 - 1)
        
        # Theta (Annualized)
        theta_base = -(F * discount_factor * N_prime_d1 * sigma) / (2 * np.sqrt(T))
        theta_call = theta_base + r * discount_factor * (F * N_d1 - K * N_d2)
        theta_put = theta_base + r * discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
        # Rho (Annualized)
        rho_call = -T * discount_factor * (F * N_d1 - K * N_d2) # Simplified
        rho_put = -T * discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1)) # Simplified
    
    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


# ==============================================================================
# 4. IMPLIED VOLATILITY AND ANALYSIS HELPERS
# ==============================================================================

def calculate_implied_volatility(market_price, option_type, F_or_S, K, T, r, q, asset_type, tolerance=1e-5, max_iterations=100):
    """Calculates Implied Volatility (IV) efficiently using the Newton-Raphson method."""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if T <= 0: return np.nan
            
        if asset_type in ['Stock', 'Index']:
            # Standard BSM floor
            floor = np.maximum(0, F_or_S * np.exp(-q * T) - K * np.exp(-r * T)) if option_type == 'Call' else \
                    np.maximum(0, K * np.exp(-r * T) - F_or_S * np.exp(-q * T))
        else: # Commodity (uses Black-76 floor)
            floor = np.maximum(0, F_or_S - K) * np.exp(-r * T) if option_type == 'Call' else \
                    np.maximum(0, K - F_or_S) * np.exp(-r * T)
                    
        # *** FIX 1: ARBITRAGE FIX (Return NaN to prevent broken Greeks) ***
        # If market price is below floor (arbitrage), IV is undefined.
        # [REPLACE THE EXISTING ARBITRAGE CHECK BLOCK WITH THIS]
        
        # *** FIX 1: ARBITRAGE FIX (Modified) ***
        # If market price is below floor (intrinsic value), IV is technically undefined.
        # Request: If difference is <= 5, clamp price to allow calculation.
        floor_diff = floor - market_price
        
        if market_price < floor - tolerance:
            if floor_diff <= 5.0:
                # Clamp market price to slightly above floor to allow Newton-Raphson to function
                # (Solver needs Price > Intrinsic to find a valid Volatility)
                market_price = floor + 0.001 
            else:
                # Gap is too large (> 5), return NaN as it is a hard arbitrage
                return np.nan


        sigma_guess = 0.20
        
        for i in range(max_iterations):
            # C3 FIX: Use BSM for Index/Stock and B-76 for Commodity
            if asset_type in ['Index', 'Stock']: 
                call_price, put_price, vega = bsm_price_and_vega_only(F_or_S, K, T, r, sigma_guess, q)
            elif asset_type == 'Commodity':
                call_price, put_price, vega = black76_price_and_vega_only(F_or_S, K, T, r, sigma_guess)
            else: return np.nan
            
            model_price = call_price if option_type == 'Call' else put_price
            price_error = model_price - market_price
            
            if abs(price_error) < tolerance: return sigma_guess
            
            # If vega is zero, we can't divide.
            if abs(vega) < 1e-9:
                if abs(price_error) < tolerance:
                    return sigma_guess
                break 
            
            sigma_guess -= price_error / vega
            
            if sigma_guess <= 0: 
                sigma_guess = tolerance 

        return np.nan # Return np.nan if no solution found


def get_moneyness_status(delta):
    """Determines the option's moneyness status based on the IV Delta."""
    if np.isnan(delta): return "N/A"
    abs_delta = abs(delta)
    
    if abs_delta >= HybridAnalysisConfig.DELTA_DEEP_ITM: 
        return "Deep ITM (High Probability)."
    elif abs_delta >= HybridAnalysisConfig.DELTA_ATM_LOWER:
        return "ATM/Near-Term Risk (50% Prob.)."
    else:
        return "OTM (Low Probability)."


# ==============================================================================
# 5. MAIN EXECUTION BLOCK (Interactive and Calculation)
# ==============================================================================

def run_options_analysis():
    print("=" * 60)
    print("Welcome to the Hybrid Options Pricing and Analysis Tool! 🤖")
    print("=" * 60)
    
    # 1. ASSET SELECTION
    asset_type = input("Enter Asset Type ('Stock', 'Index', or 'Commodity'): ").strip().title()
    if asset_type not in ['Stock', 'Index', 'Commodity']:
        print("\nError: Invalid Asset Type selected.")
        return

    asset_ticker = input(f"Enter Ticker/Code for {asset_type} (e.g., AAPL): ").strip().upper()
    
    # 2. CORE INPUTS
    try:
        print("\n--- Model Inputs ---")
        K = float(input("Enter the Strike Price (K): "))
        Days_to_expiry = float(input("Enter Time to Expiration (T, in DAYS, e.g., 30): "))
        T = Days_to_expiry / 365.0 # Ensure T is float
        
        # Handle T=0 case
        if T == 0:
            print("Warning: Time to expiry is zero. Calculations will be based on intrinsic value.")
            T = 1e-9 # Set to a tiny number to avoid division by zero
            
        r = float(input("Enter Risk-Free Rate (r, e.g., 0.05): "))
        sigma_user = float(input("Enter User Defined Volatility (σ, e.g., 0.15): "))
        
        print("\n--- Option Market Inputs ---")
        call_market = float(input("Enter Current Market - Call Option Premium (call_market): "))
        put_market = float(input("Enter Current Market - Put Option Premium (put_market): "))
        
        # 3. ASSET-SPECIFIC INPUTS
        F_or_S = 0.0
        q = 0.0
        model_q = 0.0 # q used by BSM-style models
        iv_q = 0.0 # q used by IV solver
        
        if asset_type == 'Stock' or asset_type == 'Index':
            F_or_S = float(input("Enter Current SPOT Price (S): "))
            q = float(input("Enter Dividend Yield (q, e.g., 0.005): "))
            model_q = q # BSM, Merton, Binomial use the user-provided dividend
            iv_q = q # IV solver also uses the user-provided dividend
            
        elif asset_type == 'Commodity':
            F_or_S = float(input("Enter Current FUTURES Price (F): "))
            q = 0.0 # Not applicable for commodity, but set to 0
            model_q = r # BSM, Merton, Binomial use q=r for futures
            iv_q = 0.0 # IV solver uses Black-76, which assumes q=0 (handled in function)
        
        # 3.1 ASK FOR EXTENDED MODEL INPUTS
        print("\n--- Extended Model Inputs ---")
        print("Enter Merton Jump parameters (or 0 to skip):")
        lam_merton = float(input("  Enter Jump Intensity (lambda, e.g., 1.0): "))
        mu_j_merton = float(input("  Enter Mean Log Jump Size (mu_J, e.g., -0.01): "))
        sigma_j_merton = float(input("  Enter Std Dev Log Jump Size (sigma_J, e.g., 0.05): "))
        
        print("\nEnter Numerical Model parameters:")
        N_binomial = int(input("  Enter Binomial Steps (N, e.g., 100): "))
        M_mc = int(input("  Enter Monte Carlo Paths (M, e.g., 50000): "))

        print("\n--- Option Chain Analysis Input ---")
        PCR_Ratio = float(input("Enter the CURRENT Put/Call Ratio (PCR) for this expiry: "))

        # M7 FIX: Final validation including all parameters
        validate_inputs(K, Days_to_expiry, r, sigma_user, S=(F_or_S if asset_type != 'Commodity' else None), 
                        F=(F_or_S if asset_type == 'Commodity' else None), 
                        q=(q if asset_type != 'Commodity' else None), 
                        asset_type=asset_type,
                        lam=lam_merton, mu_j=mu_j_merton, sigma_j=sigma_j_merton, N_bin=N_binomial, M_mc=M_mc)


    except ValueError as e:
        print(f"\nFATAL ERROR: Invalid Input or Arbitrage detected.\nDetails: {e}")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred during input: {e}")
        return

    # 4. EXECUTE PRICING & GREEKS (BASELINE MODELS)
    
    model_call_user, model_put_user = np.nan, np.nan
    delta_call_user, delta_put_user, gamma_user, vega_user, theta_call_user, theta_put_user, rho_call_user, rho_put_user = (np.nan,)*8
    model_call_iv, model_put_iv = np.nan, np.nan
    delta_call_iv, delta_put_iv, gamma_call_iv, gamma_put_iv, vega_call_iv, vega_put_iv, theta_call_iv, theta_put_iv, rho_call_iv, rho_put_iv = (np.nan,)*10

    baseline_model_name = "" # For output labeling

    # C3 FIX: Route to correct baseline model
    if asset_type in ['Stock', 'Index']:
        baseline_model_name = "BSM (Baseline)"
        model_call_user, model_put_user, d1_user, d2_user = black_scholes_merton(F_or_S, K, T, r, sigma_user, model_q)
        delta_call_user, delta_put_user, gamma_user, vega_user, theta_call_user, theta_put_user, rho_call_user, rho_put_user = \
            calculate_greeks_BSM(F_or_S, K, d1_user, d2_user, T, r, model_q, sigma_user)
    else: # Commodity
        baseline_model_name = "Black-76 (Baseline)"
        model_call_user, model_put_user, d1_user, d2_user = black_76(F_or_S, K, T, r, sigma_user) 
        delta_call_user, delta_put_user, gamma_user, vega_user, theta_call_user, theta_put_user, rho_call_user, rho_put_user = \
            calculate_greeks_Black76(F_or_S, K, d1_user, d2_user, T, r, sigma_user)
            
    # Calculate Daily Theta and 1% Rho for User-Sigma
    daily_theta_call_user = theta_call_user / 365
    daily_theta_put_user = theta_put_user / 365
    rho_call_percent_user = rho_call_user / 100
    rho_put_percent_user = rho_put_user / 100

    # Calculate Implied Volatilities
    IV_call = calculate_implied_volatility(call_market, 'Call', F_or_S, K, T, r, iv_q, asset_type)
    IV_put = calculate_implied_volatility(put_market, 'Put', F_or_S, K, T, r, iv_q, asset_type)
    
    # Calculate Average IV (handling NaN)
    valid_ivs = [iv for iv in [IV_call, IV_put] if not np.isnan(iv)]
    IV_avg = np.mean(valid_ivs) if valid_ivs else np.nan
    
    # Calculate IV-based Greeks (for market risk profile)
    if asset_type in ['Stock', 'Index']:
        model_call_iv, _, d1_call_iv, d2_call_iv = black_scholes_merton(F_or_S, K, T, r, IV_call, model_q)
        delta_call_iv, _, gamma_call_iv, vega_call_iv, theta_call_iv, _, rho_call_iv, _ = calculate_greeks_BSM(F_or_S, K, d1_call_iv, d2_call_iv, T, r, model_q, IV_call)
        
        _, model_put_iv, d1_put_iv, d2_put_iv = black_scholes_merton(F_or_S, K, T, r, IV_put, model_q)
        _, delta_put_iv, gamma_put_iv, vega_put_iv, _, theta_put_iv, _, rho_put_iv = calculate_greeks_BSM(F_or_S, K, d1_put_iv, d2_put_iv, T, r, model_q, IV_put)
    else: # Commodity
        model_call_iv, _, d1_call_iv, d2_call_iv = black_76(F_or_S, K, T, r, IV_call)
        delta_call_iv, _, gamma_call_iv, vega_call_iv, theta_call_iv, _, rho_call_iv, _ = calculate_greeks_Black76(F_or_S, K, d1_call_iv, d2_call_iv, T, r, IV_call)
        
        _, model_put_iv, d1_put_iv, d2_put_iv = black_76(F_or_S, K, T, r, IV_put)
        _, delta_put_iv, gamma_put_iv, vega_put_iv, _, theta_put_iv, _, rho_put_iv = calculate_greeks_Black76(F_or_S, K, d1_put_iv, d2_put_iv, T, r, IV_put)
            
    # Calculate Daily Theta and 1% Rho for IV-Sigma
    daily_theta_call_iv = theta_call_iv / 365
    daily_theta_put_iv = theta_put_iv / 365
    rho_call_percent_iv = rho_call_iv / 100
    rho_put_percent_iv = rho_put_iv / 100


    # 4.1 EXECUTE EXTENDED MODELS (PRICES & GREEKS)
    # model_q is already correctly set (user-input for Stock/Index, r for Commodity)
    
    # --- Price Lists ---
    model_names = ['Market Premium', baseline_model_name]
    call_prices = [call_market, model_call_user]
    put_prices = [put_market, model_put_user]
    
    # --- Greeks Lists ---
    greeks_model_names = [f'{baseline_model_name} (User σ)', f'{baseline_model_name} (Market IV)']
    greeks_call_list = [
        (delta_call_user, gamma_user, vega_user, daily_theta_call_user, rho_call_percent_user),
        (delta_call_iv, gamma_call_iv, vega_call_iv, daily_theta_call_iv, rho_call_percent_iv)
    ]
    greeks_put_list = [
        (delta_put_user, gamma_user, vega_user, daily_theta_put_user, rho_put_percent_user),
        (delta_put_iv, gamma_put_iv, vega_put_iv, daily_theta_put_iv, rho_put_percent_iv)
    ]

    # 1. Binomial Price & Greeks
    if N_binomial > 0:
        bin_call = binomial_pricing(F_or_S, K, T, r, sigma_user, N_binomial, 'call', True, model_q)
        bin_put = binomial_pricing(F_or_S, K, T, r, sigma_user, N_binomial, 'put', True, model_q)
        model_names.append('Binomial (American)')
        call_prices.append(bin_call)
        put_prices.append(bin_put)
        
        # Calculate Binomial Greeks
        d_c, d_p, g, v, t_c, t_p, r_c, r_p = calculate_greeks_numerical(
            binomial_pricing, F_or_S=F_or_S, K=K, T=T, r=r, sigma=sigma_user, N=N_binomial, q=model_q, is_american=True, asset_type=asset_type
        )
        greeks_model_names.append('Binomial (American)')
        greeks_call_list.append((d_c, g, v, t_c/365, r_c/100))
        greeks_put_list.append((d_p, g, v, t_p/365, r_p/100))
    
    # 2. Merton Jump-Diffusion Price & Greeks
    if lam_merton > 0 and sigma_j_merton > 0:
        try:
            merton_call = merton_jump_price(F_or_S, K, T, r, sigma_user, model_q, lam_merton, mu_j_merton, sigma_j_merton, 40, 'call')
            merton_put = merton_jump_price(F_or_S, K, T, r, sigma_user, model_q, lam_merton, mu_j_merton, sigma_j_merton, 40, 'put')
            model_names.append('Merton Jump-Diff.')
            call_prices.append(merton_call)
            put_prices.append(merton_put)
            
            # Calculate Merton Greeks
            d_c, d_p, g, v, t_c, t_p, r_c, r_p = calculate_greeks_merton(
                F_or_S=F_or_S, K=K, T=T, r=r, sigma=sigma_user, q=model_q, lam=lam_merton, mu_j=mu_j_merton, sigma_j=sigma_j_merton, asset_type=asset_type, N_jumps=40
            )
            greeks_model_names.append('Merton Jump-Diff.')
            greeks_call_list.append((d_c, g, v, t_c/365, r_c/100))
            greeks_put_list.append((d_p, g, v, t_p/365, r_p/100))

        except Exception as e:
            print(f"\nWarning: Merton model failed to calculate. {e}")
            
    # 3. Monte Carlo Price
    if M_mc > 0:
        # Note: MC is European, and Monte Carlo Greeks are skipped for computational complexity.
        mc_call = monte_carlo_price(F_or_S, K, T, r, sigma_user, model_q, M_mc, 'call')
        mc_put = monte_carlo_price(F_or_S, K, T, r, sigma_user, model_q, M_mc, 'put')
        model_names.append('Monte Carlo Sim.')
        call_prices.append(mc_call)
        put_prices.append(mc_put)

    # 5. STRUCTURED OUTPUT GENERATION
    
    # --- Input Summary ---
    input_summary = {
        'Asset': asset_ticker,
        'Asset Type': asset_type,
        'Strike Price (K)': K,
        'Time to Expiry (Days)': Days_to_expiry,
        'Risk-Free Rate (r)': f"{r*100:.2f}%",
        'User Volatility (σ)': f"{sigma_user*100:.2f}%",
        'Call Market Premium': call_market,
        'Put Market Premium': put_market,
        'Current PCR Ratio': PCR_Ratio,
        'Underlying Price (S/F)': F_or_S,
    }
    if asset_type != 'Commodity':
        input_summary['Dividend Yield (q)'] = f"{q*100:.2f}%"
    else:
        # Display the derived model q (which is r)
        input_summary['Model Dividend Yield (q)'] = f"{model_q*100:.2f}% (Set to r for Futures)" 

    input_summary['Binomial Steps (N)'] = N_binomial
    input_summary['Monte Carlo Paths (M)'] = M_mc
    input_summary['Merton Lambda (λ)'] = lam_merton
    input_summary['Merton Mu_J'] = mu_j_merton
    input_summary['Merton Sigma_J'] = sigma_j_merton

    # --- Hybrid Model Price Comparison (Request 2) ---
    model_comparison_data = {
        'Model': model_names,
        'Call Price': call_prices,
        'Put Price': put_prices
    }

    # --- FINAL SINGLE-STRIKE ANALYSIS (Request 3) ---
    # Calculate Average Theoretical Price (excluding market premium)
    avg_call_price = np.nanmean(call_prices[1:])
    avg_put_price = np.nanmean(put_prices[1:])

    # Calculate Mispricing vs. Average
    mispricing_call_avg = avg_call_price - call_market
    mispricing_put_avg = avg_put_price - put_market

    # M6: Valuation Status vs. Average
    call_val_status_avg = 'UNDERVALUED (Potential Buy)' if mispricing_call_avg > HybridAnalysisConfig.MISPRICING_BUY_THRESHOLD else \
                           'OVERVALUED (Potential Sell)' if mispricing_call_avg < HybridAnalysisConfig.MISPRICING_SELL_THRESHOLD else \
                           'FAIRLY VALUED'
    put_val_status_avg = 'UNDERVALUED (Potential Buy)' if mispricing_put_avg > HybridAnalysisConfig.MISPRICING_BUY_THRESHOLD else \
                         'OVERVALUED (Potential Sell)' if mispricing_put_avg < HybridAnalysisConfig.MISPRICING_SELL_THRESHOLD else \
                         'FAIRLY VALUED'
                         
    # Calculate Intrinsic & Time Value
    intrinsic_call = np.maximum(0, F_or_S - K)
    intrinsic_put = np.maximum(0, K - F_or_S)
    time_value_call = call_market - intrinsic_call
    time_value_put = put_market - intrinsic_put
    
    # M8: Arbitrage Check for Output
    arbitrage_flag_call = " (*)" if time_value_call < -1e-6 else ""
    arbitrage_flag_put = " (*)" if time_value_put < -1e-6 else ""
    
    # Calculate Breakeven
    breakeven_call = K + call_market
    breakeven_put = K - put_market

    # Build the new analysis table data
    analysis_table_data = {
        'Metric': [
            'Theoretical Price (Model Avg)',
            'Market Premium',
            'Intrinsic Value',
            'Time Value (Extrinsic)',
            'Theoretical Price (at IV)',
            'Mispricing (Model Avg vs Market)',
            'Valuation Status (vs Model Avg)',
            'Moneyness Status (at IV)',
            'Breakeven Price'
        ],
        'Call Option': [
            avg_call_price,
            call_market,
            intrinsic_call,
            f"{time_value_call}{arbitrage_flag_call}", # M8 FIX: Added flag
            model_call_iv,
            mispricing_call_avg,
            call_val_status_avg,
            get_moneyness_status(delta_call_iv),
            breakeven_call
        ],
        'Put Option': [
            avg_put_price,
            put_market,
            intrinsic_put,
            f"{time_value_put}{arbitrage_flag_put}", # M8 FIX: Added flag
            model_put_iv,
            mispricing_put_avg,
            put_val_status_avg,
            get_moneyness_status(delta_put_iv),
            breakeven_put
        ]
    }
    
    # --- GREEKS COMPARISON (Request 4) ---
    greeks_cols = ['Delta', 'Gamma', 'Vega', 'Theta (Day)', 'Rho (1% r)']
    
    df_greeks_call = pd.DataFrame(greeks_call_list, columns=greeks_cols)
    df_greeks_call.insert(0, 'Model', greeks_model_names)
    
    df_greeks_put = pd.DataFrame(greeks_put_list, columns=greeks_cols)
    df_greeks_put.insert(0, 'Model', greeks_model_names)


    # M5 FIX: Define custom formatter for small numbers (especially Gamma for commodities)
    def format_greek_value(x):
        if pd.isna(x): return "N/A"
        # Use scientific notation for very small numbers to avoid 0.0000
        if abs(x) < 0.0001 and abs(x) > 1e-10:
            return f"{x:.4e}" 
        # Format as standard decimal
        return f"{x:+.4f}" if x > 0 else f"{x:.4f}"
        
    df_greeks_call_formatted = df_greeks_call.copy()
    df_greeks_put_formatted = df_greeks_put.copy()
    
    # Apply custom formatting
    for col in greeks_cols:
        df_greeks_call_formatted[col] = df_greeks_call_formatted[col].apply(format_greek_value)
        df_greeks_put_formatted[col] = df_greeks_put_formatted[col].apply(format_greek_value)
        

    # ==============================================================================
    # 6. DISPLAY FINAL OUTPUT
    # ==============================================================================
    
    # --- Input Summary ---
    print("\n\n" + "=" * 60)
    print(f"| SUMMARY OF INPUTS for {asset_ticker} |")
    print("=" * 60)
    for key, value in input_summary.items():
        print(f"{key:<30}: {value}")
        
    # --- IV and Actionable Note ---
    print("\n" + "=" * 60)
    print("| IMPLIED VOLATILITY (IV) ANALYSIS |")
    print("=" * 60)
    
    # Custom formatting for NaN
    iv_call_str = f"{IV_call*100:.2f}%" if not np.isnan(IV_call) else "N/A (Arbitrage)"
    iv_put_str = f"{IV_put*100:.2f}%" if not np.isnan(IV_put) else "N/A (Arbitrage)"
    iv_avg_str = f"{IV_avg*100:.2f}%" if not np.isnan(IV_avg) else "N/A"

    print(f"User Defined Volatility (σ):      {sigma_user*100:.2f}%")
    print(f"Call Option Implied Volatilty:  {iv_call_str}")
    print(f"Put Option Implied Volatilty:   {iv_put_str}")
    
    # Actionable Note Logic
    if not np.isnan(IV_avg):
        iv_diff = IV_avg - sigma_user
        if abs(iv_diff) > HybridAnalysisConfig.IV_DIFFERENCE_PCT:
            trend = "LOWER" if iv_diff < 0 else "HIGHER"
            status = "UNDERVALUED" if iv_diff < 0 else "OVERVALUED"
            action = "Buy" if iv_diff < 0 else "Sell"
            print(f"ACTIONABLE NOTE: Market IV ({iv_avg_str}) is significantly **{trend}** than your input σ ({sigma_user*100:.2f}%).")
            print(f"This suggests the options are currently **{status}** based on your risk estimate (Potential {action}).")
        else:
            print(f"ACTIONABLE NOTE: Market IV ({iv_avg_str}) is fairly aligned with your input σ ({sigma_user*100:.2f}%).")
    else:
        # C1 FIX: Clear IV arbitrage warning
        print("ACTIONABLE NOTE: Market IV is N/A due to arbitrage pricing. Comparison to User σ is not possible.")

    print("=" * 60)

    # --- HYBRID MODEL PRICE COMPARISON ---
    print("\n\n" + "=" * 60)
    print(f"| HYBRID MODEL PRICE COMPARISON |")
    print("=" * 60)
    
    # Display the comparison DataFrame
    df_comparison = pd.DataFrame(model_comparison_data)
    print(df_comparison.to_string(index=False, float_format="{:.4f}".format))
    
    print("=" * 60)

    # --- FINAL SINGLE-STRIKE ANALYSIS (Request 3) ---
    print("\n\n" + "=" * 80)
    print(f"| FINAL SINGLE-STRIKE (PRICE VS. MODEL) ANALYSIS for {asset_ticker} |")
    print("=" * 80)

    df_analysis = pd.DataFrame(analysis_table_data)
    # Format the table for printing
    for col in ['Call Option', 'Put Option']:
        df_analysis[col] = df_analysis.apply(
            lambda row: f"₹{row[col]:,.2f}" if isinstance(row[col], (int, float)) and row['Metric'] not in ['Moneyness Status (at IV)', 'Valuation Status (vs Model Avg)']
                      else (f"{row[col]:+.2f}" if isinstance(row[col], (int, float)) and row['Metric'] == 'Mispricing (Model Avg vs Market)'
                      else (f"{row[col]:,.2f}" if isinstance(row[col], (int, float)) and row['Metric'] == 'Time Value (Extrinsic)' # <-- FIX: Apply 2 decimal formatting for Time Value      
                      else ("N/A" if isinstance(row[col], float) and np.isnan(row[col]) else row[col]))), # Handle NaNs
            axis=1
        )
    print(df_analysis.to_string(index=False))
    
    # M8 FIX: Arbitrage Note
    arbitrage_note = ""
    if arbitrage_flag_call or arbitrage_flag_put:
        arbitrage_note = ("\n(*) ARBITRAGE ALERT: Time Value is negative. This indicates the market price is below "
                          "the option's intrinsic value. This is a clear mispricing and suggests a risk-free "
                          "profit opportunity (buy the option, exercise, and sell the underlying, "
                          "or vice-versa) if transaction costs are ignored.")
    print(arbitrage_note)
    
    print("=" * 80)


    # --- GREEKS COMPARISON (Request 4) ---
    print("\n\n" + "=" * 80)
    print(f"| OPTION GREEKS COMPARISON (CALL OPTION) |")
    print("=" * 80)
    print(df_greeks_call_formatted.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"| OPTION GREEKS COMPARISON (PUT OPTION) |")
    print("=" * 80)
    print(df_greeks_put_formatted.to_string(index=False))


    # --- GREEKS INTERPRETATION & RISK ASSESSMENT ---
    print("\n\n" + "=" * 80)
    print("| GREEKS INTERPRETATION & RISK ASSESSMENT (Based on Baseline Model at Market IV) |")
    print("=" * 80)

    # m11 FIX: Calculate Gamma interpretation once
    shared_gamma_implication = HybridAnalysisConfig.get_gamma_implication(gamma_call_iv) # Use IV Gamma

    print(f"\n--- MARKET RISK PROFILE (Calculated at IV, using {baseline_model_name}) ---")
    # Call IV Analysis
    print("Call Option Risks:")
    print(f"  Gamma Risk: {shared_gamma_implication}") # m11 fix: use single calculated string
    print(f"  {HybridAnalysisConfig.get_vega_implication(vega_call_iv, F_or_S)}")
    print(f"  {HybridAnalysisConfig.get_theta_implication(daily_theta_call_iv, model_call_iv)}")
    print(f"  {HybridAnalysisConfig.get_rho_implication(rho_call_iv, F_or_S, 'Call', asset_type)}") # m10 fix: Pass asset_type
    
    # Put IV Analysis
    print("\nPut Option Risks:")
    print(f"  Gamma Risk: {shared_gamma_implication}") # m11 fix: use single calculated string
    print(f"  {HybridAnalysisConfig.get_vega_implication(vega_put_iv, F_or_S)}")
    print(f"  {HybridAnalysisConfig.get_theta_implication(daily_theta_put_iv, model_put_iv)}")
    print(f"  {HybridAnalysisConfig.get_rho_implication(rho_put_iv, F_or_S, 'Put', asset_type)}") # m10 fix: Pass asset_type

    print("\n" + "-" * 80)
    print(f"--- ESTIMATED RISK PROFILE (Calculated at User σ, using {baseline_model_name}) ---")
    # Note: Gamma and Vega are shared (same inputs: S, K, T, r, sigma_user, q)
    shared_gamma_implication_user = HybridAnalysisConfig.get_gamma_implication(gamma_user)
    
    # Call User Analysis
    print("Call Option Risks:")
    print(f"  Gamma Risk: {shared_gamma_implication_user}")
    print(f"  {HybridAnalysisConfig.get_vega_implication(vega_user, F_or_S)}")
    print(f"  {HybridAnalysisConfig.get_theta_implication(daily_theta_call_user, model_call_user)}")
    print(f"  {HybridAnalysisConfig.get_rho_implication(rho_call_user, F_or_S, 'Call', asset_type)}")
    
    # Put User Analysis
    print("\nPut Option Risks:")
    print(f"  Gamma Risk: {shared_gamma_implication_user}")
    print(f"  {HybridAnalysisConfig.get_vega_implication(vega_user, F_or_S)}")
    print(f"  {HybridAnalysisConfig.get_theta_implication(daily_theta_put_user, model_put_user)}")
    print(f"  {HybridAnalysisConfig.get_rho_implication(rho_put_user, F_or_S, 'Put', asset_type)}")

    print("=" * 80)
    
    # --- Market (PCR) Analysis ---
    print("\n\n" + "=" * 60)
    print(f"| OVERALL MARKET (OPTION CHAIN) ANALYSIS for {asset_ticker} |")
    print("=" * 60)
    
    # REVERTED TO YOUR ORIGINAL CONTRARIAN LOGIC
    sentiment = "Bullish (More Puts than Calls). The market is too bearish right now, but when everyone is hedging (buying puts), there's nobody left to sell. This exhaustion often precedes a rally (Bullish)." if PCR_Ratio > HybridAnalysisConfig.PCR_BULLISH else \
                "Bearish (More Calls than Puts). The market is too bullish right now, but when everyone is speculating on a rise (buying calls), the market is vulnerable to a drop. This often precedes a correction (Bearish)." if PCR_Ratio < HybridAnalysisConfig.PCR_BEARISH else \
                "Neutral (Balanced)"
                
    print(f"The calculated Put/Call Ratio (PCR) is: {PCR_Ratio:.2f}")
    print(f"This reflects a market sentiment that is currently: {sentiment}.")
    print("=" * 60)


# Call the function to test the execution flow
#run_options_analysis() <-- commented out to prevent execution during import


