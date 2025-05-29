def dDOdt(DO, T, pH, NH4, params):
    """
    Menghitung laju perubahan DO (dDO/dt) berdasarkan input suhu (T), pH, dan konsentrasi NH4.
    
    Parameter:
    - DO   : konsentrasi DO saat ini (mg/L)
    - T    : suhu air (°C)
    - pH   : derajat keasaman air
    - NH4  : konsentrasi amonium (mg/L)
    - params: dictionary berisi parameter model:
        * k_a20     : koefisien reaerasi pada 20°C (1/hari)
        * theta_a   : faktor suhu reaerasi
        * r_nitr    : laju konsumsi O2 per mg NH4-N (mg O2/mg NH4-N)
        * theta_n   : faktor suhu nitrifikasi
        * pK_a, pK_b: konstanta pH untuk fungsi nitrifikasi
        * mu_max    : laju maksimum fotosintesis (mg O2/L/hari)
        * theta_f   : faktor suhu fotosintesis
        * I, K_I    : intensitas cahaya dan konstanta setengah jenuh
        * A         : biomassa alga (mg/L)
        * a         : koefisien respirasi ikan
        * B         : biomassa ikan (kg/m3)
        * W         : berat rata-rata ikan (g)
    """
    # Reaerasi
    k_a = params['k_a20'] * params['theta_a']**(T - 20)
    DO_sat = 14.652 - 0.41022*T + 0.007991*T**2 - 0.000077774*T**3
    reaerasi = k_a * (DO_sat - DO)
    
    # Nitrifikasi
    fT_n = params['theta_n']**(T - 20)
    fpH = 1 / (1 + 10**(params['pK_a'] - pH)) * 1 / (1 + 10**(pH - params['pK_b']))
    nitrifikasi = params['r_nitr'] * fT_n * fpH * NH4
    
    # Fotosintesis (siang, I/(I+K_I) ≈ 1 jika I >> K_I)
    fT_f = params['theta_f']**(T - 20)
    fotosintesis = params['mu_max'] * fT_f * (params['I']/(params['I'] + params['K_I'])) * params['A']
    
    # Respirasi ikan
    respirasi = params['a'] * params['B'] * params['W']**(-0.237) * np.exp(0.063 * T)
    
    # Laju perubahan DO
    return reaerasi - nitrifikasi + fotosintesis - respirasi

# Contoh penggunaan fungsi:
params_example = {
    'k_a20': 0.5, 'theta_a': 1.024,
    'r_nitr': 4.57, 'theta_n': 1.08, 'pK_a': 6.5, 'pK_b': 9.0,
    'mu_max': 10.0, 'theta_f': 1.07, 'I': 1000.0, 'K_I': 200.0, 'A': 5.0,
    'a': 24.715, 'B': 0.5, 'W': 100.0
}

# Misal DO=5 mg/L, T=30°C, pH=8.0, NH4=1.0 mg/L:
# dDO = dDOdt(DO=5.0, T=30.0, pH=8.0, NH4=1.0, params=params_example)
# print(f"Laju perubahan DO: {dDO:.3f} mg/L/hari")
