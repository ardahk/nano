from __future__ import annotations
from flask import Flask, render_template, request, jsonify
import math
import scipy.constants
import numpy as np
from flask_cors import CORS
import pandas as pd
import json
import traceback

# Import the ebeam model
try:
    from model import ElectronBeamLithographySimulator, image_to_mask, create_2d_gaussian, create_base64_plot
    ebeam_available = True
except ImportError:
    ebeam_available = False
    print("EBeam model import failed. EBeam functionality will be disabled.")

app = Flask(__name__)
CORS(app)

# Initialize ebeam model if available
if ebeam_available:
    ebeam_model = ElectronBeamLithographySimulator()
    ebeam_display = False

# ── physical constants (rows 4-12 of the sheet) ─────────────────────────
E, kB, Na = 1.602e-19, 1.380e-23, 6.022e23
eps0, ER_WAT, ZVAL, PI = 8.854e-12, 78.49, 1, math.pi

# ── defaults copied from the clean sheet ───────────────────────────────
DEFAULTS = {
    # geometry (uses 10 µm × 10 µm pad, 100 µm CNT height)
    "cnt_radius":   5e-9,
    "cnt_height":   1e-4,
    "base_width":   1e-5,
    "base_length":  1e-5,
    "cnt_gap":      0.0,
    "base_thickness": 5.5412e-4,

    # stack thickness rows (display only)
    "t_si":     3e-4,  "t_sio2_1": 2.1e-6,  "t_sio2_2": 1e-6,
    "t_ti":     1e-7,  "t_al2o3":  1e-8,    "t_al":    1e-8,
    "t_fe":     1e-8,

    # functional layer
    "mol_radius": 5e-10,
    "mol_length": 2e-9,
    "mol_gap": 1e-9,
    "er_mol":     8.0,

    # electrolyte / dielectric
    "conc_molar":      10.0,
    "temperature":     298.1,
    "zeta_potential":  0.05,
    "er_solvent":      ER_WAT,
    "freq_stage1":     1e5,   # 100 kHz
    "freq_stage2":     1e4,   # 10 kHz
    "series_resistance": 1e5,

    # op-amp parameters
    "Rf": 1e5, "Vref": 1.65, "Vmax": 3.3, "Vmin": 0.0,

    # rows that exist in the UI but aren't used in the math
    "energy_density": "", "voltage": "", "max_vout": "",
    "concentration_standard": "", "ion_species_change": "",
    "charging_current": "", "charging_voltage": "", "lens_resistivity": "",
    "lens_density": "", "lens_specific_heat": "", "lens_height": "",
    "lens_area": "", "lens_r_in": "", "lens_r_out": "", "lithography_time": "",
}

_f = lambda v, d: d if v in ("", None) else float(v)

# ── clean-sheet equations ──────────────────────────────────────────────
def simulate(raw: dict) -> dict:
    p = {k: _f(raw.get(k), v) for k, v in DEFAULTS.items()}

    # EDL differential capacitance / area
    C0   = p["conc_molar"] * 1e3
    cosh = math.cosh(ZVAL * E * p["zeta_potential"] /
                     (2 * kB * p["temperature"]))
    C_EDL_A = math.sqrt(
        2 * eps0 * p["er_solvent"] * Na * C0 * (ZVAL * E) ** 2 /
        (kB * p["temperature"])
    ) * cosh

    # CNT array surface area
    Sa_cnt = PI * p["cnt_radius"] ** 2 + 2 * PI * p["cnt_radius"] * p["cnt_height"]
    n_cnt  = (p["base_width"] * p["base_length"]) / (PI * (2 * p["cnt_radius"]) ** 2)
    Sa_tot = Sa_cnt * n_cnt

    C_EDL = C_EDL_A * Sa_tot
    C_H   = eps0 * p["er_mol"] / p["mol_length"]            * Sa_tot
    C_Hp  = C_H
    #C_Hp  = eps0 * p["er_mol"] / (p["mol_length"] + p["mol_radius"]) * Sa_tot

    # Stage 1  (EDL + H)
    C1 = 1 / (1 / C_EDL + 1 / C_H)
    Z1 = 1 / (2 * PI * p["freq_stage1"] * C1)
    I1 = p["zeta_potential"] / Z1
    Vo1 = p["Rf"] * abs(2 * p["freq_stage1"] * C1 * (p["Vmax"] - p["Vmin"]))

    # Stage 2  (EDL + H + H′)
    C2 = 1 / (1 / C_EDL + 1 / C_H + 1 / C_Hp)
    Z2 = 1 / (2 * PI * p["freq_stage2"] * C2)
    I2 = p["zeta_potential"] / Z2
    Vo2 = p["Rf"] * abs(2 * p["freq_stage2"] * C2 * (p["Vmax"] - p["Vmin"]))

    return {
        "stage1": {"C": C1, "Z": Z1, "I": I1, "Vo": Vo1},
        "stage2": {"C": C2, "Z": Z2, "I": I2, "Vo": Vo2},
        "Delta_I": I1 - I2,
        "Delta_Z": Z1 - Z2,
    }

# Define a function to add navigation links to the template context
def add_navigation_context(context=None):
    if context is None:
        context = {}
    context['nav_links'] = [
        {'url': '/', 'text': 'Home'},
        {'url': '/sheet1', 'text': 'Sheet 1'},
        {'url': '/vo_rf', 'text': 'VO RF'},
        {'url': '/vx', 'text': 'VX'},
        {'url': '/covid', 'text': 'COVID Detection'},
        {'url': '/biosensing', 'text': 'Theoretical Simulation'},
        {'url': '/bio_project', 'text': 'Project Management'}
    ]
    return context

@app.route("/")
def index():
    context = add_navigation_context()
    return render_template("index.html", **context)

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
    context = add_navigation_context()
    return render_template("sheet1.html", **context)

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
    context = add_navigation_context()
    return render_template("vo_rf.html", **context)

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
    context = add_navigation_context()
    return render_template("vx.html", **context)

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
    context = add_navigation_context()
    return render_template("covid.html", **context)

@app.route("/calculate_covid", methods=["POST"])
def calculate_covid():
    """
    implementation of the "COVID Detection Clean Sheet".
    All formulas come straight from the Excel.Only a handful of
    inputs are required; everything else defaults to the sheet's constants
    but may be overridden by the front‑end
    """
    data = request.get_json()

    # inputs
    epsilon_r = float(data.get("epsilon_r", 78.49))      # relative permittivity – cell B6
    z         = float(data.get("z", 1))                  # ion valence           – cell B3
    V_zeta    = float(data.get("V_zeta", 0.05))          # zeta potential (V)    – cell B10
    # bulk concentration (mol m⁻³) – cell B5.  
    # Excel sheet uses 10 mM, which is 10 000 mol m^-3 in SI units.
    C0        = float(data.get("C0", 10000))
    freq1     = float(data.get("freq", 100_000))         # Hz, branch 1          – cell B32
    freq2     = float(data.get("freq2", 10_000))         # Hz, branch 2          – cell B44

    # CNT geometry (may be overridden)
    r_CNT     = float(data.get("r_CNT", 5e-9))           # radius  – cell B20
    h_CNT     = float(data.get("h_CNT", 1e-4))           # height  – cell B21

    # chip/platform size – defaults give the 1 mm × mm used in the sheet
    chip_L    = float(data.get("chip_L", 1e-3))          # m  – cell B23
    chip_W    = float(data.get("chip_W", 1e-3))          # m  – cell B24

    #gap distances that set the Helmholtz capacitances CH and CH1
    gap_d1    = float(data.get("gap_d1", 2e-8))          # m  = around 19.6 nm in the sheet
    gap_d2    = float(data.get("gap_d2", 2e-8))          # m  (same for CH1)

    er_DNA = float(data.get("er_DNA", 8))     # cell B39
    L_DNA  = float(data.get("L_DNA", 2e-9))   # cell B40

    # constants 
    eps0 = scipy.constants.epsilon_0
    e_ch = scipy.constants.e
    k_B  = scipy.constants.k
    T    = 298.1                                           # K (room temp in sheet)
    N_A  = scipy.constants.Avogadro
    PI   = math.pi

    #  derived CNT & sensor geometry 
    SA_1_CNT = 2 * PI * r_CNT * h_CNT                      # cell B22
    CNT_count = (chip_L * chip_W) / SA_1_CNT               # cell B25

    # electro‑double‑layer 
    # Debye length (cell B14)
    d_EDL = math.sqrt(epsilon_r * eps0 * k_B * T / (2 * (z * e_ch) ** 2 * N_A * C0))

    # C_EDL for one CNT (cell B13)
    pre_factor = math.sqrt(2 * (z * e_ch) ** 2 * N_A * C0 * epsilon_r * eps0 / (k_B * T))
    cosh_term = math.cosh((z * e_ch * V_zeta) / (2 * k_B * T))  # sheet col D12
    CEDL_1_CNT = SA_1_CNT * pre_factor * cosh_term

    # C_EDL for the entire sensor (cell B17)
    CEDL_total = CEDL_1_CNT * CNT_count

    # ───────────────────── Helmholtz double‑layer (CH) and DNA hybridization layer (CH1) ──────────────
    CH_per_area  = eps0 * epsilon_r / gap_d1     # cell B? (original CH per area)
    CH           = CH_per_area * SA_1_CNT * CNT_count
    CH1_per_area = eps0 * er_DNA      / L_DNA    # cell F39‑F40 (DNA-layer cap)
    CH1          = CH1_per_area * SA_1_CNT * CNT_count

    #  Branch 1 (no CH1) 
    inv_C1 = (1 / CH) + (1 / CEDL_total)                    # cell B30
    C1     = 1 / inv_C1                                     # cell B31
    Z1     = 1 / (2 * PI * freq1 * C1)                      # cell B33
    I1     = V_zeta / Z1                                    # cell B35

    #fixed‑resistor path (R = 100 kΩ) 
    R_fixed = float(data.get("R_fixed", 1e5))             # cell E34 in sheet
    Z_RC    = math.sqrt(R_fixed ** 2 + (1 / (2 * PI * freq1 * C1)) ** 2)
    I_RC    = V_zeta / Z_RC                               # optional current through RC branch

    # Branch 2 (with CH1) 
    inv_C2 = (1 / CH) + (1 / CH1) + (1 / CEDL_total)        # cell B42
    C2     = 1 / inv_C2                                     # cell B43
    Z2     = 1 / (2 * PI * freq2 * C2)                      # cell B45
    I2     = V_zeta / Z2                                    # cell B47

    # Deltas – match sign convention in the Excel sheet
    delta_I = abs(I1 - I2)          # cell B50  (unchanged magnitude)
    delta_Z = Z1 - Z2               # cell B51  (negative because impedance rises when CH/CH1 are added)
    delta_C = C1 - C2               # cell B52  (positive because total capacitance falls when CH/CH1 are added)

    # extras to mirror every Excel field 
    # Per‑unit‑area and per‑CNT capacitances
    C_EDL_per_area = CEDL_1_CNT / SA_1_CNT                # cell B12 (F m^-2)
    # CH_per_area and CH1_per_area updated above
    CH_per_CNT     = CH  / CNT_count
    CH1_per_CNT    = CH1 / CNT_count

    # Inverse capacitances that appear explicitly on the sheet
    inv_C1 = 1 / C1                                       # cell B30
    inv_C2 = 1 / C2                                       # cell B42

    # Fundamental constants shown on the sheet
    F_eNA   = scipy.constants.e * N_A                     # Faraday constant (≈ 96485 C mol^-1)
    R_kNa   = k_B * N_A                                   # Universal gas constant (≈ 8.314 J mol^-1 K^-1)

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

        # Branch 1
        "C_total_branch1 (F)":   f"{C1:.3e}",
        "Z_branch1 (Ω)":         f"{Z1:.3e}",
        "I_branch1 (A)":         f"{I1:.3e}",
        "Z_RC (Ω)":               f"{Z_RC:.3e}",
        "I_with_R (A)":           f"{I_RC:.3e}",

        # Branch 2
        "C_total_branch2 (F)":   f"{C2:.3e}",
        "Z_branch2 (Ω)":         f"{Z2:.3e}",
        "I_branch2 (A)":         f"{I2:.3e}",

        # Differences
        "ΔI (A)":                f"{delta_I:.3e}",
        "ΔZ (Ω)":                f"{delta_Z:.3e}",
        "ΔC (F)":                f"{delta_C:.3e}",
    })

@app.route("/simulate", methods=["POST"])
def run_simulation():
    """Process the simulation for the biosensing page (from app1.py)"""
    return jsonify(simulate(request.get_json(force=True)))

@app.route("/biosensing")
def biosensing():
    """Show the biosensing theoretical simulation page"""
    try:
        return render_template("Biosensing/TheoreticalSimulation1.html", defaults=DEFAULTS)
    except Exception as e:
        app.logger.error(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/run_ebeam_simulation', methods=['POST'])
def run_ebeam_simulation():
    """Process the ebeam simulation and return results as a CSV file"""
    if not ebeam_available:
        return jsonify({"status": "error", "error": "EBeam functionality is not available"}), 500
        
    try:
        # Get the mask data from the request
        data = request.get_json()
        mask = data.get('mask', None)
        if mask is None:
            return jsonify({"status": "error", "error": "No mask data provided"}), 400
            
        # Run the simulation using the ebeam model
        print("Running ebeam simulation...")
        json_result = ebeam_model.run_lithography(mask=mask, display=ebeam_display)
        
        # Extract the dataframe - EXACTLY as in the original code
        df_json = json_result.get("df_json")
        df_json = json.loads(df_json)
        df = pd.DataFrame(data=df_json["data"], columns=df_json["columns"], index=df_json["index"])
        print(df)
        csv_data = df.to_csv(index=False)
        
        # Return success with CSV data - EXACTLY as in the original code
        return jsonify(
            status="success",
            csv_data=csv_data
        )
    except Exception as e:
        print(f"Error in ebeam simulation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Simulation error: {str(e)}"}), 500

@app.route("/calculate_biosensing", methods=["POST"])
def calculate_biosensing():
    """
    Legacy/alternative calculation endpoint for biosensing simulation
    """
    data = request.get_json()
    
    try:
        # Convert the data format to match what the simulate function expects
        simulation_data = {
            "cnt_radius": float(data.get("radius_cnt", DEFAULTS["cnt_radius"])),
            "cnt_height": float(data.get("height_cnt", DEFAULTS["cnt_height"])),
            "base_width": float(data.get("width_base", DEFAULTS["base_width"])),
            "base_length": float(data.get("length_base", DEFAULTS["base_length"])),
            "cnt_gap": float(data.get("gap_cnts", DEFAULTS["cnt_gap"])),
            "base_thickness": float(data.get("thickness_base", DEFAULTS["base_thickness"])),
            
            "mol_radius": float(data.get("radius_molecule", DEFAULTS["mol_radius"])),
            "mol_length": float(data.get("length_molecule", DEFAULTS["mol_length"])),
            "mol_gap": float(data.get("gap_molecules", DEFAULTS["mol_gap"])),
            "er_mol": float(data.get("dielectric_molecule", DEFAULTS["er_mol"])),
            
            "conc_molar": float(data.get("c_std_state", DEFAULTS["conc_molar"])),
            "er_solvent": float(data.get("rel_perm_solvent", DEFAULTS["er_solvent"])),
            "temperature": float(data.get("temp_solvent", DEFAULTS["temperature"])),
            "zeta_potential": float(data.get("zeta_potential", DEFAULTS["zeta_potential"])),
            "freq_stage1": float(data.get("frequency", DEFAULTS["freq_stage1"])),
            "freq_stage2": float(data.get("frequency", DEFAULTS["freq_stage2"])) / 10, # If only one frequency provided, use 1/10th for stage 2
            "series_resistance": float(data.get("r_solution", DEFAULTS["series_resistance"]))
        }
        
        # Run the simulation with the converted data
        results = simulate(simulation_data)
        
        # Format the results to match the expected legacy format
        legacy_results = {
            "cnt_surface_area": 2 * PI * simulation_data["cnt_radius"] * simulation_data["cnt_height"],
            "base_area": simulation_data["base_width"] * simulation_data["base_length"],
            "stage1_capacitance": results["stage1"]["C"],
            "stage1_impedance": results["stage1"]["Z"],
            "stage1_current": results["stage1"]["I"],
            "stage2_capacitance": results["stage2"]["C"],
            "stage2_impedance": results["stage2"]["Z"],
            "stage2_current": results["stage2"]["I"],
            "delta_current": results["Delta_I"],
            "delta_impedance": results["Delta_Z"]
        }
        
        return jsonify(legacy_results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/bio_project")
def bio_project():
    """Show the biosensing project management page"""
    context = add_navigation_context()
    return render_template("Biosensing/BioProjectManagment.html", **context)

# ---- EBeam Integration ----
@app.route('/ebeam')
def ebeam():
    return render_template("ebeam/index.html")

@app.route('/update_params', methods=['POST'])
def update_params():
    """
    This updates the parameter of the underlying model to avoid re-initialization.
    """
    if not ebeam_available:
        return jsonify({"error": "EBeam functionality is not available"}), 500
        
    global ebeam_display

    # Log original parameters
    print("\n=== Prev Model Parameters ===")
    print(f"Wafer Dimensions: {ebeam_model.wafer_dim}")
    print(f"CNT Grid Shape: {ebeam_model.cnt_grid_shape}")
    print(f"CNT Unit Dimensions: {ebeam_model.cnt_unit_dim}")
    print(f"Display: {ebeam_display}")

    # extract the data parameters parameters
    data = request.get_json()
    params = data.get('params', {})
    wafer_dim = tuple(params.get('wafer_dim', ebeam_model.wafer_dim))
    cnt_grid_shape = tuple(params.get('cnt_grid_shape', ebeam_model.cnt_grid_shape))
    cnt_unit_dim = tuple(params.get('cnt_unit_dim', ebeam_model.cnt_unit_dim))

    # update display
    ebeam_display = bool(params['display'])

    # update model
    ebeam_model.update_parameters(wafer_dim, cnt_grid_shape, cnt_unit_dim)

    # Log updated parameters
    print("\n=== Updated Model Parameters ===")
    print(f"Wafer Dimensions: {ebeam_model.wafer_dim}")
    print(f"CNT Grid Shape: {ebeam_model.cnt_grid_shape}")
    print(f"CNT Unit Dimensions: {ebeam_model.cnt_unit_dim}")
    print(f"Display: {ebeam_display}")

    # return success
    return jsonify(status="success")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5003)