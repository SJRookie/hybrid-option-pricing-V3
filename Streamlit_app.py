
"""
===============================================================================
INTERACTIVE OPTION PRICING DASHBOARD
Built with Streamlit & Plotly
===============================================================================
Author: Converted for Streamlit Deployment
Version: 2.0
Description: Professional web-based option pricing and Greeks analysis tool
             with multi-model support (BSM, Black-76, Merton, Binomial, MC)
===============================================================================
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Import all pricing functions from your core module
# Make sure option_pricing_core.py is in the same directory
from option_pricing_core import (
    validate_inputs, black_scholes_merton, black_76,
    calculate_greeks_BSM, calculate_greeks_Black76,
    calculate_greeks_merton, calculate_greeks_numerical,
    binomial_pricing, merton_jump_price, monte_carlo_price,
    calculate_implied_volatility, get_moneyness_status,
    HybridAnalysisConfig
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# HELPER FUNCTIONS (for Streamlit UI)
# ============================================================================

def create_valuation_display(mispricing_value):
    """Create color-coded markdown based on mispricing threshold"""
    if pd.isna(mispricing_value):
        return "<span style='color:gray;'>N/A</span>"

    if mispricing_value > HybridAnalysisConfig.MISPRICING_BUY_THRESHOLD:
        return f"<span style='color:green; font-weight:bold;'>UNDERVALUED (Potential Buy) ↑ (₹{mispricing_value:+.2f})</span>"
    elif mispricing_value < HybridAnalysisConfig.MISPRICING_SELL_THRESHOLD:
        return f"<span style='color:red; font-weight:bold;'>OVERVALUED (Potential Sell) ↓ (₹{mispricing_value:+.2f})</span>"
    else:
        return f"<span style='color:gray; font-weight:bold;'>FAIRLY VALUED (₹{mispricing_value:+.2f})</span>"

def format_risk_text(text_block):
    """Converts newlines to <br> tags for HTML rendering in markdown"""
    if pd.isna(text_block): return "N/A"
    return text_block.replace("\n", "<br>")

def format_greek_value(value):
    """Format Greek values for display in tables"""
    if pd.isna(value):
        return "N/A"
    if abs(value) < 0.0001 and abs(value) > 1e-10:
        return f"{value:.4e}"
    return f"{value:+.4f}" if value != 0 else f"{value:.4f}"

def create_payoff_chart(r):
    """Generates the P/L payoff chart"""
    K = r['strike']
    S_range = np.linspace(K * 0.7, K * 1.3, 100) # Adjusted range

    call_pl = np.maximum(S_range - K, 0) - r['call_market']
    put_pl = np.maximum(K - S_range, 0) - r['put_market']

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Call P/L (Breakeven: ₹{r["breakeven_call"]:.2f})', 
                                                       f'Put P/L (Breakeven: ₹{r["breakeven_put"]:.2f})'))

    # Call Plot
    fig.add_trace(go.Scatter(x=S_range, y=call_pl, mode='lines', name='Call',
                            line=dict(color='blue', width=2), fill='tozeroy'), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_vline(x=K, line_dash="dot", line_color="gray", row=1, col=1, annotation_text="Strike")
    fig.add_vline(x=r['underlying'], line_color="orange", row=1, col=1, annotation_text="S/F")
    fig.add_vline(x=r['breakeven_call'], line_dash="dash", line_color="blue", row=1, col=1, annotation_text="B/E")


    # Put Plot
    fig.add_trace(go.Scatter(x=S_range, y=put_pl, mode='lines', name='Put',
                            line=dict(color='red', width=2), fill='tozeroy'), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="green", row=1, col=2)
    fig.add_vline(x=K, line_dash="dot", line_color="gray", row=1, col=2, annotation_text="Strike")
    fig.add_vline(x=r['underlying'], line_color="orange", row=1, col=2, annotation_text="S/F")
    fig.add_vline(x=r['breakeven_put'], line_dash="dash", line_color="red", row=1, col=2, annotation_text="B/E")

    fig.update_xaxes(title_text="Price at Expiration", row=1, col=1)
    fig.update_xaxes(title_text="Price at Expiration", row=1, col=2)
    fig.update_yaxes(title_text="P/L (₹)", row=1, col=1)
    fig.update_yaxes(title_text="P/L (₹)", row=1, col=2)
    fig.update_layout(height=450, showlegend=False, template='plotly_white')
    return fig

def create_greek_chart(results, greek_name, title_name):
    """
    *** NEW FEATURE FUNCTION ***
    Dynamically creates a bar chart for any given Greek.
    """
    models = results['greeks_model_names']
    # Ensure values are floats for plotting, handle N/A
    call_vals = [g[greek_name] if not pd.isna(g[greek_name]) else 0.0 for g in results['greeks_call_list']]
    put_vals = [g[greek_name] if not pd.isna(g[greek_name]) else 0.0 for g in results['greeks_put_list']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Call', x=models, y=call_vals, marker_color='blue',
        text=[f"{v:.4f}" for v in call_vals], textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Put', x=models, y=put_vals, marker_color='red',
        text=[f"{v:.4f}" for v in put_vals], textposition='outside'
    ))
    
    # Add a horizontal line at 0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")

    fig.update_layout(
        title=f"{title_name} Comparison",
        xaxis_title="Model",
        yaxis_title=title_name,
        barmode='group',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ============================================================================
# SIDEBAR (INPUT PANEL)
# ============================================================================

st.sidebar.header("Model Parameters")

# Row 1
asset_type = st.sidebar.selectbox(
    "Asset Type",
    ['Stock', 'Index', 'Commodity'],
    index=1,
    help="Select the underlying asset type. This determines the pricing model (BSM vs Black-76)."
)
ticker = st.sidebar.text_input("Ticker", "NIFTY")
K = st.sidebar.number_input("Strike (K)", value=24000.0, step=50.0)

# Row 2
T_days = st.sidebar.number_input("Days to Expiry", value=30, min_value=1)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.065, format="%.4f")
sigma_user = st.sidebar.number_input("User Volatility (σ)", value=0.18, format="%.4f")

# Row 3
call_market = st.sidebar.number_input("Call Premium", value=450.50, format="%.2f")
put_market = st.sidebar.number_input("Put Premium", value=280.75, format="%.2f")
S_or_F = st.sidebar.number_input("Spot/Futures (S/F)", value=24150.0, step=1.0)

# Row 4
is_commodity = (asset_type == 'Commodity')
q_help = "Set to 0 for Commodities, as Black-76 (q=r) is used." if is_commodity else "Annual dividend yield."
q = st.sidebar.number_input("Dividend (q)", value=0.0, format="%.4f", disabled=is_commodity, help=q_help)
PCR = st.sidebar.number_input("PCR Ratio", value=1.15, format="%.2f")

# Advanced Parameters
with st.sidebar.expander("Advanced Model Inputs"):
    st.markdown("_(Set to 0 to skip a model)_")
    
    c1, c2, c3 = st.columns(3)
    lam = c1.number_input("λ (Jump)", value=0.0, format="%.2f")
    mu_j = c2.number_input("μ_J", value=0.0, format="%.2f")
    sigma_j = c3.number_input("σ_J", value=0.0, format="%.2f")
    
    c4, c5 = st.columns(2)
    N_bin = c4.number_input("Binomial Steps (N)", value=100, step=10)
    M_mc = c5.number_input("MC Paths (M)", value=50000, step=1000)

# Calculate Button
calculate_button = st.sidebar.button("Calculate", type="primary", use_container_width=True)

# ============================================================================
# MAIN CALCULATION & PAGE RENDER
# ============================================================================

st.title(f"📈 Option Pricing Dashboard: {ticker}")

if calculate_button:
    try:
        # 1. VALIDATE INPUTS
        validate_inputs(K=K, T_days=T_days, r=r, sigma=sigma_user,
                       S=(S_or_F if asset_type != 'Commodity' else None),
                       F=(S_or_F if asset_type == 'Commodity' else None),
                       q=(q if asset_type != 'Commodity' else None),
                       asset_type=asset_type,
                       lam=lam, mu_j=mu_j, sigma_j=sigma_j,
                       N_bin=N_bin, M_mc=M_mc)
        
        # 2. PREPARE PARAMETERS
        T = T_days / 365.0
        model_q = q if asset_type != 'Commodity' else r # q=r for Black-76/Futures
        iv_q = q if asset_type != 'Commodity' else 0.0 # B-76 IV calc uses q=0
        
        results = {
            'asset_type': asset_type, 'ticker': ticker, 'strike': K,
            'time_days': T_days, 'underlying': S_or_F, 'dividend': q,
            'call_market': call_market, 'put_market': put_market,
            'sigma_user': sigma_user, 'rate': r, 'pcr': PCR, 'model_q': model_q
        }
        
        # 3. BASELINE MODEL (User Sigma)
        if asset_type in ['Stock', 'Index']:
            call_u, put_u, d1, d2 = black_scholes_merton(S_or_F, K, T, r, sigma_user, model_q)
            dc, dp, g, v, tc, tp, rc, rp = calculate_greeks_BSM(S_or_F, K, d1, d2, T, r, model_q, sigma_user)
            baseline = "BSM"
        else:
            call_u, put_u, d1, d2 = black_76(S_or_F, K, T, r, sigma_user)
            dc, dp, g, v, tc, tp, rc, rp = calculate_greeks_Black76(S_or_F, K, d1, d2, T, r, sigma_user)
            baseline = "Black-76"

        results['baseline_model'] = baseline
        results['call_user'] = call_u
        results['put_user'] = put_u
        results['greeks_user'] = {
            'delta_call': dc, 'delta_put': dp, 'gamma': g, 'vega': v,
            'theta_call': tc/365, 'theta_put': tp/365,
            'rho_call': rc/100, 'rho_put': rp/100
        }
        
        # 4. IMPLIED VOLATILITY
        IV_call = calculate_implied_volatility(call_market, 'Call', S_or_F, K, T, r, iv_q, asset_type)
        IV_put = calculate_implied_volatility(put_market, 'Put', S_or_F, K, T, r, iv_q, asset_type)
        results['IV_call'] = IV_call
        results['IV_put'] = IV_put
        results['IV_avg'] = np.nanmean([IV_call, IV_put])
        
        # 5. BASELINE MODEL (Market IV)
        if asset_type in ['Stock', 'Index']:
            c_iv, _, d1c, d2c = black_scholes_merton(S_or_F, K, T, r, IV_call, model_q)
            dc_iv, _, gc, vc, tc_iv, _, rc_iv, _ = calculate_greeks_BSM(S_or_F, K, d1c, d2c, T, r, model_q, IV_call)
            _, p_iv, d1p, d2p = black_scholes_merton(S_or_F, K, T, r, IV_put, model_q)
            _, dp_iv, gp, vp, _, tp_iv, _, rp_iv = calculate_greeks_BSM(S_or_F, K, d1p, d2p, T, r, model_q, IV_put)
        else:
            c_iv, _, d1c, d2c = black_76(S_or_F, K, T, r, IV_call)
            dc_iv, _, gc, vc, tc_iv, _, rc_iv, _ = calculate_greeks_Black76(S_or_F, K, d1c, d2c, T, r, IV_call)
            _, p_iv, d1p, d2p = black_76(S_or_F, K, T, r, IV_put)
            _, dp_iv, gp, vp, _, tp_iv, _, rp_iv = calculate_greeks_Black76(S_or_F, K, d1p, d2p, T, r, IV_put)

        results['call_iv'] = c_iv
        results['put_iv'] = p_iv
        results['greeks_iv'] = {
            'delta_call': dc_iv, 'delta_put': dp_iv,
            'gamma_call': gc, 'gamma_put': gp,
            'vega_call': vc, 'vega_put': vp,
            'theta_call': tc_iv/365, 'theta_put': tp_iv/365,
            'rho_call': rc_iv/100, 'rho_put': rp_iv/100
        }
        
        # 6. BUILD MODEL LISTS
        model_names = ['Market', baseline]
        call_prices = [call_market, call_u]
        put_prices = [put_market, put_u]

        greeks_models = [f'{baseline} (User σ)', f'{baseline} (Market IV)']
        greeks_call = [
            {'Delta': dc, 'Gamma': g, 'Vega': v, 'Theta': tc/365, 'Rho': rc/100},
            {'Delta': dc_iv, 'Gamma': gc, 'Vega': vc, 'Theta': tc_iv/365, 'Rho': rc_iv/100}
        ]
        greeks_put = [
            {'Delta': dp, 'Gamma': g, 'Vega': v, 'Theta': tp/365, 'Rho': rp/100},
            {'Delta': dp_iv, 'Gamma': gp, 'Vega': vp, 'Theta': tp_iv/365, 'Rho': rp_iv/100}
        ]
        
        # 7. ADVANCED MODELS
        
        # Binomial
        if N_bin > 0:
            try:
                bc = binomial_pricing(S_or_F, K, T, r, sigma_user, N_bin, 'call', True, model_q)
                bp = binomial_pricing(S_or_F, K, T, r, sigma_user, N_bin, 'put', True, model_q)
                model_names.append('Binomial')
                call_prices.append(bc)
                put_prices.append(bp)

                dc2, dp2, g2, v2, tc2, tp2, rc2, rp2 = calculate_greeks_numerical(
                    binomial_pricing, S_or_F, K, T, r, sigma_user, N_bin, model_q, True, asset_type)
                greeks_models.append('Binomial')
                greeks_call.append({'Delta': dc2, 'Gamma': g2, 'Vega': v2, 'Theta': tc2/365, 'Rho': rc2/100})
                greeks_put.append({'Delta': dp2, 'Gamma': g2, 'Vega': v2, 'Theta': tp2/365, 'Rho': rp2/100})
            except Exception as e:
                st.warning(f"Binomial model failed: {e}")

        # Merton
        if lam > 0 and sigma_j > 0:
            try:
                mc = merton_jump_price(S_or_F, K, T, r, sigma_user, model_q, lam, mu_j, sigma_j, 40, 'call')
                mp = merton_jump_price(S_or_F, K, T, r, sigma_user, model_q, lam, mu_j, sigma_j, 40, 'put')
                model_names.append('Merton')
                call_prices.append(mc)
                put_prices.append(mp)

                dc3, dp3, g3, v3, tc3, tp3, rc3, rp3 = calculate_greeks_merton(
                    S_or_F, K, T, r, sigma_user, model_q, lam, mu_j, sigma_j, asset_type, 40)
                greeks_models.append('Merton')
                greeks_call.append({'Delta': dc3, 'Gamma': g3, 'Vega': v3, 'Theta': tc3/365, 'Rho': rc3/100})
                greeks_put.append({'Delta': dp3, 'Gamma': g3, 'Vega': v3, 'Theta': tp3/365, 'Rho': rp3/100})
            except Exception as e:
                st.warning(f"Merton model failed: {e}")

        # Monte Carlo
        if M_mc > 0:
            try:
                mcc = monte_carlo_price(S_or_F, K, T, r, sigma_user, model_q, M_mc, 'call')
                mcp = monte_carlo_price(S_or_F, K, T, r, sigma_user, model_q, M_mc, 'put')
                model_names.append('Monte Carlo')
                call_prices.append(mcc)
                put_prices.append(mcp)
            except Exception as e:
                st.warning(f"Monte Carlo model failed: {e}")
        
        # 8. FINALIZE RESULTS
        results['model_names'] = model_names
        results['call_prices'] = call_prices
        results['put_prices'] = put_prices
        results['greeks_model_names'] = greeks_models
        results['greeks_call_list'] = greeks_call
        results['greeks_put_list'] = greeks_put

        # Analysis
        results['avg_call'] = np.nanmean(call_prices[1:])
        results['avg_put'] = np.nanmean(put_prices[1:])
        results['mispricing_call'] = results['avg_call'] - call_market
        results['mispricing_put'] = results['avg_put'] - put_market
        results['intrinsic_call'] = max(0, S_or_F - K)
        results['intrinsic_put'] = max(0, K - S_or_F)
        results['time_value_call'] = call_market - results['intrinsic_call']
        results['time_value_put'] = put_market - results['intrinsic_put']
        results['breakeven_call'] = K + call_market
        results['breakeven_put'] = K - put_market
        results['moneyness_call'] = get_moneyness_status(dc_iv)
        results['moneyness_put'] = get_moneyness_status(dp_iv)
        
        # Store in session state
        st.session_state['results'] = results
        st.success("Calculation Successful!")
        
    except ValueError as e:
        st.error(f"Input Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# ============================================================================
# TABS (OUTPUT DISPLAY)
# ============================================================================

if 'results' not in st.session_state:
    st.info("Please enter parameters in the sidebar and click 'Calculate'.")
    st.stop()

# Load results from state
r = st.session_state['results']

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Greeks", "⚠️ Risk Analysis", "📉 Visualizations"])

# ----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------------------
with tab1:
    # --- Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asset", f"{r['ticker']} ({r['asset_type']})")
    col2.metric("Strike", f"₹{r['strike']:,.0f}")
    col3.metric("Days to Expiry", f"{r['time_days']:.0f}")
    col4.metric("Underlying (S/F)", f"₹{r['underlying']:,.0f}")

    # --- Arbitrage Alert ---
    if r['time_value_call'] < -1e-6:
        st.error(f"Arbitrage Alert (Call): Market price (₹{r['call_market']:.2f}) is below intrinsic value (₹{r['intrinsic_call']:.2f})!")
    if r['time_value_put'] < -1e-6:
        st.error(f"Arbitrage Alert (Put): Market price (₹{r['put_market']:.2f}) is below intrinsic value (₹{r['intrinsic_put']:.2f})!")

    col1, col2 = st.columns(2)
    
    with col1:
        # --- Price Comparison Table ---
        st.subheader("Price Comparison")
        prices_df = pd.DataFrame({
            'Model': r['model_names'],
            'Call': r['call_prices'],
            'Put': r['put_prices']
        })
        st.dataframe(
            prices_df.style.format({'Call': '₹{:.2f}', 'Put': '₹{:.2f}'})
                           .apply(lambda x: ['background-color: #008080; font-weight: bold;' if x.name == 0 else '' for i in x], axis=1),
            use_container_width=True, 
            hide_index=True
        )
        
        # --- Market (PCR) Analysis ---
        st.subheader("Market (PCR) Analysis")
        pcr = r['pcr']
        if pcr > HybridAnalysisConfig.PCR_BULLISH:
            sent = ("📈 **Contrarian Bullish**: The higher PCR suggests heavy Put buying (hedging or bearish bets). Often, when everyone is hedged, the market is positioned for a rise.", "success", "Excessive Put buying")
            sent_color = "green"
        elif pcr < HybridAnalysisConfig.PCR_BEARISH:
            sent = ("📉 **Contrarian Bearish**: The lower PCR suggests heavy Call buying (speculation). Excessive bullishness can make the market vulnerable to a correction.", "danger", "Excessive Call buying")
            sent_color = "red"
        else:
            sent = ("⚖️ **Neutral**: Balanced put-call ratio indicates market equilibrium or indecision.", "secondary", "Balanced activity")
            sent_color = "gray"
        
        st.metric("PCR Ratio", f"{pcr:.2f}", sent[2])
        st.markdown(f"**Sentiment: <span style='color:{sent_color};'>{sent[0]}</span>**", unsafe_allow_html=True)
        
    with col2:
        # --- Single-Strike Analysis ---
        st.subheader("Single-Strike Analysis")
        analysis_data = [
            ['Theoretical (Avg)', f"₹{r['avg_call']:.2f}", f"₹{r['avg_put']:.2f}"],
            ['Market Premium', f"₹{r['call_market']:.2f}", f"₹{r['put_market']:.2f}"],
            ['Mispricing (Avg vs Mkt)', f"{r['mispricing_call']:+.2f}", f"{r['mispricing_put']:+.2f}"],
            ['Intrinsic Value', f"₹{r['intrinsic_call']:.2f}", f"₹{r['intrinsic_put']:.2f}"],
            ['Time Value (Extrinsic)', f"₹{r['time_value_call']:.2f}", f"₹{r['time_value_put']:.2f}"],
            ['Breakeven Price', f"₹{r['breakeven_call']:.2f}", f"₹{r['breakeven_put']:.2f}"],
            ['Implied Volatility (IV)', f"{r['IV_call']*100:.2f}%" if not pd.isna(r['IV_call']) else "N/A",
             f"{r['IV_put']*100:.2f}%" if not pd.isna(r['IV_put']) else "N/A"],
            ['Price at IV', f"₹{r['call_iv']:.2f}" if not pd.isna(r['call_iv']) else "N/A", 
             f"₹{r['put_iv']:.2f}" if not pd.isna(r['put_iv']) else "N/A"],
            ['Moneyness (at IV)', r['moneyness_call'], r['moneyness_put']],
        ]
        analysis_df = pd.DataFrame(analysis_data, columns=['Metric', 'Call', 'Put'])
        st.dataframe(analysis_df, use_container_width=True, hide_index=True)

        # --- Valuation Badges ---
        st.markdown(f"**Call Valuation:** {create_valuation_display(r['mispricing_call'])}", unsafe_allow_html=True)
        st.markdown(f"**Put Valuation:** {create_valuation_display(r['mispricing_put'])}", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 2: GREEKS
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Greeks Comparison Tables")
    
    # Format Call Greeks
    df_c = pd.DataFrame(r['greeks_call_list'])
    df_c.insert(0, 'Model', r['greeks_model_names'])
    greeks_to_format = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    for col in greeks_to_format:
        if col in df_c.columns:
            df_c[col] = df_c[col].apply(format_greek_value)
            
    st.markdown("#### Call Greeks")
    st.dataframe(df_c, use_container_width=True, hide_index=True)
    
    # Format Put Greeks
    df_p = pd.DataFrame(r['greeks_put_list'])
    df_p.insert(0, 'Model', r['greeks_model_names'])
    for col in greeks_to_format:
         if col in df_p.columns:
            df_p[col] = df_p[col].apply(format_greek_value)

    st.markdown("#### Put Greeks")
    st.dataframe(df_p, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 3: RISK ANALYSIS
# ----------------------------------------------------------------------------
with tab3:
    giv = r['greeks_iv']
    gu = r['greeks_user']
    S = r['underlying']
    at = r['asset_type']

    st.subheader(f"Market Risk (Based on IV: {r['IV_avg']*100:.2f}%)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Call Option Risks (at IV)")
        st.markdown(f"**Delta:** {format_risk_text(HybridAnalysisConfig.get_delta_implication(giv['delta_call'], 'Call'))}", unsafe_allow_html=True)
        st.markdown(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(giv['gamma_call'])}")
        st.markdown(f"**Vega:** {format_risk_text(HybridAnalysisConfig.get_vega_implication(giv['vega_call'], S))}", unsafe_allow_html=True)
        st.markdown(f"**Theta:** {format_risk_text(HybridAnalysisConfig.get_theta_implication(giv['theta_call'], r['call_iv']))}", unsafe_allow_html=True)
        st.markdown(f"**Rho:** {format_risk_text(HybridAnalysisConfig.get_rho_implication(giv['rho_call']*100, S, 'Call', at))}", unsafe_allow_html=True)
    with col2:
        st.markdown("#### Put Option Risks (at IV)")
        st.markdown(f"**Delta:** {format_risk_text(HybridAnalysisConfig.get_delta_implication(giv['delta_put'], 'Put'))}", unsafe_allow_html=True)
        st.markdown(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(giv['gamma_put'])}")
        st.markdown(f"**Vega:** {format_risk_text(HybridAnalysisConfig.get_vega_implication(giv['vega_put'], S))}", unsafe_allow_html=True)
        st.markdown(f"**Theta:** {format_risk_text(HybridAnalysisConfig.get_theta_implication(giv['theta_put'], r['put_iv']))}", unsafe_allow_html=True)
        st.markdown(f"**Rho:** {format_risk_text(HybridAnalysisConfig.get_rho_implication(giv['rho_put']*100, S, 'Put', at))}", unsafe_allow_html=True)
        
    st.divider()

    st.subheader(f"Estimated Risk (Based on User σ: {r['sigma_user']*100:.2f}%)")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Call Option Risks (at User σ)")
        st.markdown(f"**Delta:** {format_risk_text(HybridAnalysisConfig.get_delta_implication(gu['delta_call'], 'Call'))}", unsafe_allow_html=True)
        st.markdown(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(gu['gamma'])}")
        st.markdown(f"**Vega:** {format_risk_text(HybridAnalysisConfig.get_vega_implication(gu['vega'], S))}", unsafe_allow_html=True)
        st.markdown(f"**Theta:** {format_risk_text(HybridAnalysisConfig.get_theta_implication(gu['theta_call'], r['call_user']))}", unsafe_allow_html=True)
        st.markdown(f"**Rho:** {format_risk_text(HybridAnalysisConfig.get_rho_implication(gu['rho_call']*100, S, 'Call', at))}", unsafe_allow_html=True)
    with col4:
        st.markdown("#### Put Option Risks (at User σ)")
        st.markdown(f"**Delta:** {format_risk_text(HybridAnalysisConfig.get_delta_implication(gu['delta_put'], 'Put'))}", unsafe_allow_html=True)
        st.markdown(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(gu['gamma'])}")
        st.markdown(f"**Vega:** {format_risk_text(HybridAnalysisConfig.get_vega_implication(gu['vega'], S))}", unsafe_allow_html=True)
        st.markdown(f"**Theta:** {format_risk_text(HybridAnalysisConfig.get_theta_implication(gu['theta_put'], r['put_user']))}", unsafe_allow_html=True)
        st.markdown(f"**Rho:** {format_risk_text(HybridAnalysisConfig.get_rho_implication(gu['rho_put']*100, S, 'Put', at))}", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 4: VISUALIZATIONS
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Profit/Loss Payoff at Expiration")
    st.plotly_chart(create_payoff_chart(r), use_container_width=True)
    
    st.divider()
    
    st.subheader("Greeks Comparison Visuals")
    
    # Use the new helper function to create all 5 charts
    st.plotly_chart(create_greek_chart(r, 'Delta', 'Delta'), use_container_width=True)
    st.plotly_chart(create_greek_chart(r, 'Gamma', 'Gamma'), use_container_width=True)
    st.plotly_chart(create_greek_chart(r, 'Vega', 'Vega'), use_container_width=True)
    st.plotly_chart(create_greek_chart(r, 'Theta', 'Theta (Daily)'), use_container_width=True)

    st.plotly_chart(create_greek_chart(r, 'Rho', 'Rho (per 1% rate change)'), use_container_width=True)

