from flask import Flask, render_template, request, jsonify
import math

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
if __name__ == "__main__":
    app.run(debug=True)