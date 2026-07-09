# =============================================================================
# APPLICATION WEB — PROFIL LITHIASIQUE PAK
# Prototype Flask : le patient entre ses valeurs biologiques,
# l'application lui retourne son profil phénotypique + recommandations
# Auteur  : Nafissa Boumanjal — M2 ISC — CHU Amiens — Dir. Dr. Brazier
# Lancer  : python app.py  →  ouvrir http://localhost:5000
# =============================================================================

from flask import Flask, render_template, request, jsonify   # Bibliothèques Flask
import numpy as np                                             # Calculs numériques
import warnings                                                # Masque les avertissements
warnings.filterwarnings("ignore")                              # Désactive les warnings Python

app = Flask(__name__)   # Crée l'application Flask

# =============================================================================
# DEFINITION DES 4 PROFILS CLINIQUES
# Ces profils sont basés sur les résultats du clustering K-Means (CHU Amiens)
# Chaque profil correspond à un cluster identifié dans la cohorte de 1157 patients
# =============================================================================
PROFILS = {
    0: {
        "nom": "Homme âgé polycomorbide",                  # Nom du profil
        "description": "Profil caractérisé par une hypertension artérielle, "
                       "un surpoids ou une obésité, et un début d'insuffisance "
                       "rénale. Ce profil concerne principalement des hommes âgés "
                       "avec plusieurs facteurs de risque métaboliques associés.",
        "couleur": "#E74C3C",                                  # Couleur rouge pour l'interface
        "icone": "🫀",                                          # Icône représentative
        "recommandations": [                                    # Recommandations cliniques validées Dr. Brazier
            "Contrôle tensionnel régulier (objectif < 130/80 mmHg)",
            "Surveillance de la fonction rénale (créatinine, DFG) tous les 6 mois",
            "Réduction des apports en sodium (< 6g/jour)",
            "Hydratation abondante (objectif > 2L d'urine/jour)",
            "Contrôle du diabète si présent (HbA1c < 7%)",
            "Consultation néphrologique recommandée",
        ],
        "signes_alerte": [                                      # Signes d'alerte à surveiller
            "Douleur lombaire intense (colique néphrétique)",
            "Diminution du volume urinaire",
            "Pression artérielle > 160/100 mmHg",
        ]
    },
    1: {
        "nom": "Femme jeune sans facteur de risque majeur",
        "description": "Profil le plus fréquent chez les femmes jeunes, sans "
                       "comorbidité métabolique majeure. La lithiase est souvent "
                       "liée à une déshydratation chronique ou à des facteurs "
                       "alimentaires. Bon pronostic avec une prise en charge hygiéno-diététique.",
        "couleur": "#2ECC71",
        "icone": "💧",
        "recommandations": [
            "Hydratation optimale : 2 à 2.5L d'eau par jour minimum",
            "Répartir la consommation d'eau tout au long de la journée",
            "Surveiller la couleur des urines (objectif : jaune clair)",
            "Limiter les aliments très riches en oxalates (épinards, betteraves, chocolat)",
            "Bilan biologique annuel de contrôle",
            "Activité physique régulière recommandée",
        ],
        "signes_alerte": [
            "Douleur lombaire aiguë (colique néphrétique)",
            "Sang dans les urines (hématurie)",
            "Fièvre avec douleur lombaire (urgence médicale)",
        ]
    },
    2: {
        "nom": "Homme jeune obèse hypercalciurique",
        "description": "Profil caractérisé par un surpoids ou une obésité abdominale "
                       "avec une excrétion urinaire de calcium élevée. Ce profil est "
                       "associé à un risque plus élevé de calculs calciques oxaliques. "
                       "La perte de poids est un objectif thérapeutique prioritaire.",
        "couleur": "#8E44AD",
        "icone": "⚖️",
        "recommandations": [
            "Objectif de perte de poids progressif (5-10% du poids corporel)",
            "Réduction des apports en protéines animales (< 0.8g/kg/jour)",
            "Apport calcique alimentaire normal (ne pas supprimer le calcium alimentaire)",
            "Réduction des sucres rapides et des graisses saturées",
            "Hydratation : 2.5L d'eau/jour minimum",
            "Activité physique adaptée : 30 min/jour minimum",
            "Contrôle de la calciurie des 24h tous les 6 mois",
        ],
        "signes_alerte": [
            "Colique néphrétique récidivante",
            "Calciurie > 0.1 mmol/kg/jour à surveiller avec le médecin",
            "Prise de poids rapide",
        ]
    },
    3: {
        "nom": "Homme jeune hyperuricémique",
        "description": "Profil caractérisé par un taux d'acide urique sanguin élevé "
                       "avec un BMI globalement normal. Ce profil est associé à la "
                       "lithiase urique, favorisée par une alimentation riche en purines "
                       "(viandes, abats, fruits de mer). L'alcalinisation des urines "
                       "est souvent nécessaire.",
        "couleur": "#E67E22",
        "icone": "🧬",
        "recommandations": [
            "Réduction des aliments riches en purines (viandes rouges, abats, charcuterie, fruits de mer)",
            "Limiter la consommation d'alcool (notamment la bière)",
            "Hydratation abondante : > 2.5L/jour pour diluer l'acide urique",
            "Alcalinisation des urines si prescrite par le médecin (pH urinaire cible > 6)",
            "Contrôle de l'uricémie tous les 6 mois",
            "Traitement médicamenteux si uricémie > 600 µmol/L (à décider avec le médecin)",
        ],
        "signes_alerte": [
            "Crise de goutte (douleur articulaire soudaine, notamment au gros orteil)",
            "Uricémie > 600 µmol/L",
            "Colique néphrétique",
        ]
    }
}

# =============================================================================
# FONCTION DE CLASSIFICATION
# Détermine le profil d'un patient à partir de ses valeurs biologiques
# Méthode simplifiée basée sur les règles de l'arbre de décision PyXAI
# =============================================================================
def classifier_patient(donnees):
    """
    Classifie un patient dans l'un des 4 profils selon ses valeurs biologiques.
    Utilise les règles dérivées de l'arbre de décision PyXAI (précision 77.9%).

    Args:
        donnees (dict): Dictionnaire contenant les valeurs biologiques du patient

    Returns:
        tuple: (numero_profil, score_confiance, variables_manquantes)
    """
    score = {0: 0, 1: 0, 2: 0, 3: 0}     # Score de ressemblance à chaque profil
    variables_manquantes = []               # Variables manquantes pour la classification

    # --- Règle 1 : Sexe ---
    sexe = donnees.get("sexe")                      # Récupère le sexe
    if sexe is None:
        variables_manquantes.append("Sexe")          # Signale que le sexe manque
    else:
        if sexe == "femme":                           # Si c'est une femme
            score[1] += 3                              # Favorise fortement le profil 1 (femme jeune saine)
        else:                                          # Si c'est un homme
            score[0] += 1                               # Légèrement vers profil 0
            score[2] += 1                               # Légèrement vers profil 2
            score[3] += 1                               # Légèrement vers profil 3

    # --- Règle 2 : Age ---
    age = donnees.get("age")                         # Récupère l'âge
    if age is None:
        variables_manquantes.append("Age")
    else:
        if age >= 65:                                 # Si patient âgé (>=65 ans)
            score[0] += 3                              # Favorise fortement le profil 0 (homme âgé)
        else:                                          # Si patient jeune
            score[1] += 1                               # Légèrement vers profil 1
            score[2] += 1                               # Légèrement vers profil 2
            score[3] += 1                               # Légèrement vers profil 3

    # --- Règle 3 : HTA ---
    hta = donnees.get("hta")                         # Récupère la présence d'HTA
    if hta is not None:
        if hta:                                        # Si HTA présente
            score[0] += 3                               # Favorise fortement le profil 0

    # --- Règle 4 : BMI ---
    bmi = donnees.get("bmi")                         # Récupère le BMI
    if bmi is None:
        variables_manquantes.append("BMI")
    else:
        if bmi >= 29:                                 # Si obèse (BMI >= 29)
            score[2] += 3                              # Favorise fortement le profil 2 (obèse)
        elif bmi < 27:                                 # Si poids normal
            score[3] += 2                               # Favorise le profil 3 (hyperuricémique)

    # --- Règle 5 : Uricémie ---
    uricemie = donnees.get("uricemie")               # Récupère l'uricémie (µmol/L)
    if uricemie is not None:
        if uricemie > 360:                            # Si uricémie élevée (> 360 µmol/L)
            score[3] += 3                              # Favorise fortement le profil 3
            if bmi is not None and bmi >= 29:          # Si aussi obèse
                score[2] += 2                           # Favorise aussi le profil 2
        elif uricemie < 300:                           # Si uricémie basse
            score[1] += 2                               # Favorise le profil 1 (femme saine)

    # --- Règle 6 : Calciurie ---
    calciurie = donnees.get("calciurie")             # Récupère la calciurie des 24h (mmol/24h)
    if calciurie is not None:
        if calciurie > 3.0:                           # Si calciurie élevée (> 3.0 mmol/24h)
            score[2] += 2                              # Favorise le profil 2 (hypercalciurique)

    # --- Règle 7 : DFG ---
    dfg = donnees.get("dfg")                         # Récupère le DFG (mL/min)
    if dfg is not None:
        if dfg < 75:                                  # Si DFG bas (insuffisance rénale débutante)
            score[0] += 2                              # Favorise le profil 0 (homme âgé)

    # --- Règle 8 : Périmètre abdominal ---
    perimetre = donnees.get("perimetre")             # Récupère le périmètre abdominal (cm)
    if perimetre is not None:
        if perimetre > 100:                           # Si obésité abdominale (> 100 cm)
            score[0] += 1                              # Légèrement vers profil 0
            score[2] += 2                               # Favorise le profil 2

    # --- Classification finale : prend le profil avec le score le plus élevé ---
    profil_final = max(score, key=score.get)          # Profil avec le score maximum
    score_total = sum(score.values())                  # Score total pour calculer la confiance
    confiance = round(score[profil_final] / max(score_total, 1) * 100)  # % de confiance

    return profil_final, confiance, variables_manquantes   # Retourne le profil, la confiance, les manquants

# =============================================================================
# ROUTES FLASK
# =============================================================================

@app.route("/")                          # Route principale : page d'accueil
def index():
    """Affiche la page d'accueil avec le formulaire de saisie"""
    return render_template("index.html")  # Retourne le template HTML

@app.route("/analyser", methods=["POST"])   # Route d'analyse : reçoit les données du formulaire
def analyser():
    """Reçoit les valeurs biologiques, classe le patient, retourne les résultats"""
    try:
        # Récupère les données du formulaire HTML (envoyées en POST)
        donnees = {
            "sexe":       request.form.get("sexe"),                   # Sexe du patient
            "age":        float(request.form.get("age", 0) or 0),    # Age en années
            "hta":        request.form.get("hta") == "oui",           # HTA oui/non
            "diabete":    request.form.get("diabete") == "oui",       # Diabète oui/non
            "bmi":        float(request.form.get("bmi", 0) or 0) or None,       # BMI kg/m²
            "uricemie":   float(request.form.get("uricemie", 0) or 0) or None,  # Uricémie µmol/L
            "calciurie":  float(request.form.get("calciurie", 0) or 0) or None, # Calciurie mmol/24h
            "dfg":        float(request.form.get("dfg", 0) or 0) or None,       # DFG mL/min
            "perimetre":  float(request.form.get("perimetre", 0) or 0) or None, # Périmètre abdo cm
        }

        # Classe le patient dans l'un des 4 profils
        profil_num, confiance, manquants = classifier_patient(donnees)
        profil = PROFILS[profil_num]   # Récupère les informations du profil

        # Retourne les résultats en JSON pour affichage dynamique
        return jsonify({
            "succes": True,
            "profil_num": profil_num,
            "nom": profil["nom"],
            "description": profil["description"],
            "couleur": profil["couleur"],
            "icone": profil["icone"],
            "recommandations": profil["recommandations"],
            "signes_alerte": profil["signes_alerte"],
            "confiance": confiance,
            "manquants": manquants,
        })

    except Exception as e:
        return jsonify({"succes": False, "erreur": str(e)})   # Retourne l'erreur si problème

@app.route("/profils")                   # Route pour afficher tous les profils
def tous_profils():
    """Affiche la page avec la description des 4 profils"""
    return render_template("profils.html", profils=PROFILS)   # Passe les profils au template

# =============================================================================
# LANCEMENT DE L'APPLICATION
# =============================================================================
if __name__ == "__main__":
    print("="*55)
    print("  Application PAK Lithiase — CHU Amiens")
    print("  Ouvrir dans le navigateur : http://localhost:5000")
    print("="*55)
    app.run(debug=True, port=5000)   # Lance le serveur Flask sur le port 5000
