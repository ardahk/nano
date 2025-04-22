from flask import Flask, render_template, request, jsonify
import math
import scipy.constants
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    try:
        # get inputs
        r = float(data.get("r", 0))
        h = float(data.get("h", 0))
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
    except Exception as e:
        return jsonify({"error"}), 400

    if r <= 0:
        return jsonify({"error"}), 400

    # calculate area
    cnt_surface_area = 2 * math.pi * r * h

    # base area
    base_area = x * y

    # count CNTs
    num_cnt = base_area / (math.pi * r**2)

    # total area
    total_surface_area = num_cnt * cnt_surface_area

    # capacitance pair
    factor1 = 4.33e6
    c_pair = factor1 * cnt_surface_area

    # chip capacitance
    factor2 = 5040
    c_chip = c_pair * factor2

    # energy chip
    V = 1000
    energy_chip = 0.5 * c_chip * V**2

    # energy Wh
    wh = energy_chip / 54000

    # energy mAh
    mah = wh * 333

    # prepare results
    result = {
        "cnt_surface_area": cnt_surface_area,
        "base_area": base_area,
        "num_cnt": num_cnt,
        "total_surface_area": total_surface_area,
        "c_pair": c_pair,
        "c_chip": c_chip,
        "energy_chip": energy_chip,
        "wh": wh,
        "mah": mah,
        "voltage_assumed": V
    }
    return jsonify(result)

@app.route("/sheet1")
def sheet1():
    return render_template("sheet1.html")

@app.route("/calculate_sheet1", methods=["POST"])
def calculate_sheet1():
    data = request.get_json()
    try:
        # get inputs
        f    = float(data.get("f",    0.25))
        C    = float(data.get("C",    0.000096))
        R    = float(data.get("R",    98))
        Vfix = float(data.get("Vfix", 1.65))
        Vin  = float(data.get("Vin",  0.3))
        Rf   = float(data.get("Rf",   10000))
    except Exception as e:
        return jsonify({"error"}), 400

    PI = 22/7
    T  = 1.0 / f if f != 0 else None
    two_pi_f_C = 2 * PI * f * C
    Xc = 1.0 / two_pi_f_C if two_pi_f_C != 0 else None
    R2_plus_Xc2 = R**2 + (Xc or 0)**2
    Z  = math.sqrt(R2_plus_Xc2)
    i  = Vin / Z if Z != 0 else None

    # calculate outputs
    Vwork = Vfix + Vin
    Vout  = Vwork + (i or 0) * Rf

    # get limits
    Vmax = abs(Vin)
    Vmin = -abs(Vin)
    amplitude = abs(Vin)

    return jsonify({
        "T": T,
        "two_pi_f_C": two_pi_f_C,
        "Xc": Xc,
        "R2_plus_Xc2": R2_plus_Xc2,
        "Z": Z,
        "i": i,
        "Vwork": Vwork,
        "Vout": Vout,
        "Vmax": Vmax,
        "Vmin": Vmin,
        "amplitude": amplitude
    })
# ────────────────────────────────────────────────────────────
#  vo and rf page
# ────────────────────────────────────────────────────────────
@app.route("/vo_rf")
def vo_rf():
    """show page"""
    return render_template("vo_rf.html")

@app.route("/calculate_vo_rf", methods=["POST"])
def calculate_vo_rf():
    """
    calculate branches
    """
    d = request.get_json()

    # get parameters
    f      = float(d.get("f",      0.25))     # Hz
    Vmax   = float(d.get("Vmax",   1.0))      # V
    Vmin   = float(d.get("Vmin",  -1.0))      # V
    R      = float(d.get("R",      100))      # Ω
    C      = float(d.get("C",    1e-4))       # F
    Rf     = float(d.get("Rf",    1000))      # Ω
    t      = float(d.get("t",        2))      # s

    Vref   = float(d.get("Vref",  1.65))      # V
    PI = 22/7        

    # derived values
    Vpeak  = (Vmax - Vmin) / 2            
    Xc     = 1 / (2 * PI * f * C)
    Z      = math.sqrt(R**2 + Xc**2)

    # peak current
    i_mag = Vpeak / Z     

    def branch(sign):      
        slope = sign * 2 * f * Vpeak
        Vwork = slope * t + Vref
        i     = -sign * i_mag    
        Vo    = Vwork + i * Rf
        return {"Vwork": Vwork, "i": i, "Vo": Vo}
    down = branch(-1)  
    up   = branch(+1)  

    return jsonify({
        "Vpeak": Vpeak,
        "Xc": Xc,
        "Z": Z,
        "i_mag": i_mag,
        "i_peak": i_mag,
        "down": down,
        "up": up
    })
# ────────────────────────────────────────────────────────────
#  vx page
# ────────────────────────────────────────────────────────────
@app.route("/vx")
def vx():
    """show page"""
    return render_template("vx.html")

@app.route("/calculate_vx", methods=["POST"])
def calculate_vx():
    """
    calculate points
    """
    d = request.get_json()
    Vmax = float(d.get("Vmax",  1.0))
    Vmin = float(d.get("Vmin", -1.0))
    f    = float(d.get("f",    0.25))   

    Vpeak = (Vmax - Vmin) / 2           
    t_vals  = [0, 0.25, 0.5, 0.75, 1]   
    vx_vals = [Vpeak * t for t in t_vals]

    return jsonify({"t": t_vals, "vx": vx_vals,
                    "Vpeak": Vpeak, "f": f})

@app.route("/covid")
def covid():
    """show covid detection page"""
    return render_template("covid.html")

@app.route("/calculate_covid", methods=["POST"])
def calculate_covid():
    """
    implementation of the “COVID Detection Clean Sheet”.
    All formulas come straight from the Excel.Only a handful of
    inputs are required; everything else defaults to the sheet’s constants
    but may be overridden by the front‑end
    """
    data = request.get_json()

    # inputs
    epsilon_r = float(data.get("epsilon_r", 78.49))      # relative permittivity – cell B6
    z         = float(data.get("z", 1))                  # ion valence           – cell B3
    V_zeta    = float(data.get("V_zeta", 0.05))          # zeta potential (V)    – cell B10
    # bulk concentration (mol m⁻³) – cell B5.  
    # Excel sheet uses 10 mM, which is 10 000 mol m^-3 in SI units.
    C0        = float(data.get("C0", 10000))
    freq1     = float(data.get("freq", 100_000))         # Hz, branch 1          – cell B32
    freq2     = float(data.get("freq2", 10_000))         # Hz, branch 2          – cell B44

    # CNT geometry (may be overridden)
    r_CNT     = float(data.get("r_CNT", 5e-9))           # radius  – cell B20
    h_CNT     = float(data.get("h_CNT", 1e-4))           # height  – cell B21

    # chip/platform size – defaults give the 1 mm × 1 mm used in the sheet
    chip_L    = float(data.get("chip_L", 1e-3))          # m  – cell B23
    chip_W    = float(data.get("chip_W", 1e-3))          # m  – cell B24

    #gap distances that set the Helmholtz capacitances CH and CH1
    gap_d1    = float(data.get("gap_d1", 2e-8))          # m  = around 19.6 nm in the sheet
    gap_d2    = float(data.get("gap_d2", 2e-8))          # m  (same for CH1)

    er_DNA = float(data.get("er_DNA", 8))     # cell B39
    L_DNA  = float(data.get("L_DNA", 2e-9))   # cell B40

    # constants 
    eps0 = scipy.constants.epsilon_0
    e_ch = scipy.constants.e
    k_B  = scipy.constants.k
    T    = 298.1                                           # K (room temp in sheet)
    N_A  = scipy.constants.Avogadro
    PI   = math.pi

    #  derived CNT & sensor geometry 
    SA_1_CNT = 2 * PI * r_CNT * h_CNT                      # cell B22
    CNT_count = (chip_L * chip_W) / SA_1_CNT               # cell B25

    # electro‑double‑layer 
    # Debye length (cell B14)
    d_EDL = math.sqrt(epsilon_r * eps0 * k_B * T / (2 * (z * e_ch) ** 2 * N_A * C0))

    # C_EDL for one CNT (cell B13)
    pre_factor = math.sqrt(2 * (z * e_ch) ** 2 * N_A * C0 * epsilon_r * eps0 / (k_B * T))
    cosh_term = math.cosh((z * e_ch * V_zeta) / (2 * k_B * T))  # sheet col D12
    CEDL_1_CNT = SA_1_CNT * pre_factor * cosh_term

    # C_EDL for the entire sensor (cell B17)
    CEDL_total = CEDL_1_CNT * CNT_count

    # ───────────────────── Helmholtz double‐layer (CH) and DNA hybridization layer (CH1) ──────────────
    CH_per_area  = eps0 * epsilon_r / gap_d1     # cell B? (original CH per area)
    CH           = CH_per_area * SA_1_CNT * CNT_count
    CH1_per_area = eps0 * er_DNA      / L_DNA    # cell F39‑F40 (DNA-layer cap)
    CH1          = CH1_per_area * SA_1_CNT * CNT_count

    #  Branch 1 (no CH1) 
    inv_C1 = (1 / CH) + (1 / CEDL_total)                    # cell B30
    C1     = 1 / inv_C1                                     # cell B31
    Z1     = 1 / (2 * PI * freq1 * C1)                      # cell B33
    I1     = V_zeta / Z1                                    # cell B35

    #fixed‑resistor path (R = 100 kΩ) 
    R_fixed = float(data.get("R_fixed", 1e5))             # cell E34 in sheet
    Z_RC    = math.sqrt(R_fixed ** 2 + (1 / (2 * PI * freq1 * C1)) ** 2)
    I_RC    = V_zeta / Z_RC                               # optional current through RC branch

    # Branch 2 (with CH1) 
    inv_C2 = (1 / CH) + (1 / CH1) + (1 / CEDL_total)        # cell B42
    C2     = 1 / inv_C2                                     # cell B43
    Z2     = 1 / (2 * PI * freq2 * C2)                      # cell B45
    I2     = V_zeta / Z2                                    # cell B47

    # Deltas – match sign convention in the Excel sheet
    delta_I = abs(I1 - I2)          # cell B50  (unchanged magnitude)
    delta_Z = Z1 - Z2               # cell B51  (negative because impedance rises when CH/CH1 are added)
    delta_C = C1 - C2               # cell B52  (positive because total capacitance falls when CH/CH1 are added)

    # extras to mirror every Excel field 
    # Per‑unit‑area and per‑CNT capacitances
    C_EDL_per_area = CEDL_1_CNT / SA_1_CNT                # cell B12 (F m^-2)
    # CH_per_area and CH1_per_area updated above
    CH_per_CNT     = CH  / CNT_count
    CH1_per_CNT    = CH1 / CNT_count

    # Inverse capacitances that appear explicitly on the sheet
    inv_C1 = 1 / C1                                       # cell B30
    inv_C2 = 1 / C2                                       # cell B42

    # Fundamental constants shown on the sheet
    F_eNA   = scipy.constants.e * N_A                     # Faraday constant (≈ 96485 C mol^-1)
    R_kNa   = k_B * N_A                                   # Universal gas constant (≈ 8.314 J mol^-1 K^-1)

    #  Response 
    return jsonify({
        # Geometry
        "SA_1_CNT (m^2)":        f"{SA_1_CNT:.3e}",
        "CNT_count":             f"{CNT_count:.3f}",

        # Double‑layer
        "Debye Length d_EDL (m)":f"{d_EDL:.3e}",
        "C_EDL_one_CNT (F)":     f"{CEDL_1_CNT:.3e}",
        "C_EDL_total (F)":       f"{CEDL_total:.3e}",

        # Helmholtz caps
        "CH (F)":                f"{CH:.3e}",
        "CH1 (F)":               f"{CH1:.3e}",

        # Sheet reference values
        "C_EDL/A (F/m^2)":        f"{pre_factor:.3e}",   # cell B12
        "cosh_term":              f"{cosh_term:.3e}",   # cell D12
        "F = e·N_A (C/mol)":      f"{F_eNA:.3f}",       # cell B18
        "R = k·N_A (J/mol·K)":    f"{R_kNa:.3f}",       # cell B19

        "Z_CEDL_total (Ω)":       f"{(1 / (2 * PI * freq2 * CEDL_total)):.3e}",

        # Per‑unit‑area/per‑CNT caps
        "C_EDL_per_area (F/m^2)":     f"{C_EDL_per_area:.3e}",
        "CH_per_area (F/m^2)":        f"{CH_per_area:.3e}",
        "CH1_per_area (F/m^2)":       f"{CH1_per_area:.3e}",
        "C_EDL_per_CNT (F)":          f"{CEDL_1_CNT:.3e}",   # alias for clarity
        "CH_per_CNT (F)":             f"{CH_per_CNT:.3e}",
        "CH1_per_CNT (F)":            f"{CH1_per_CNT:.3e}",

        # Inverse caps
        "1/C_branch1 (1/F)":          f"{inv_C1:.3e}",
        "1/C_branch2 (1/F)":          f"{inv_C2:.3e}",

        # Branch 1
        "C_total_branch1 (F)":   f"{C1:.3e}",
        "Z_branch1 (Ω)":         f"{Z1:.3e}",
        "I_branch1 (A)":         f"{I1:.3e}",
        "Z_RC (Ω)":               f"{Z_RC:.3e}",
        "I_with_R (A)":           f"{I_RC:.3e}",

        # Branch 2
        "C_total_branch2 (F)":   f"{C2:.3e}",
        "Z_branch2 (Ω)":         f"{Z2:.3e}",
        "I_branch2 (A)":         f"{I2:.3e}",

        # Differences
        "ΔI (A)":                f"{delta_I:.3e}",
        "ΔZ (Ω)":                f"{delta_Z:.3e}",
        "ΔC (F)":                f"{delta_C:.3e}",
    })

if __name__ == "__main__":
    app.run(debug=True)