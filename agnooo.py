from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vérification de la clé API
if not GEMINI_API_KEY:
    raise ValueError("La clé API GEMINI_API_KEY n'est pas définie dans les variables d'environnement.")

# Création de l'agent avec le modèle Gemini et contrôle de la température
agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.7),
    description="Tu es un assistant médical spécialisé en allergologie respiratoire.",
    instructions=[
        "Lis attentivement le Logigramme.",
        "Analyse la discussion entre le médecin et le patient.",
        "Prends en compte la documentation fournie.",
        "Utilise tes données personnelles et la documentation fournie pour approfondir les petits détails oubliés par le médecin.",
        "Propose une seule question pertinente, pas trop longue, à poser au patient en fonction des informations disponibles."
    ], 
    markdown=True
)

logigramme = """
**Logigramme pour le diagnostic des allergies respiratoires**
1. **L'interrogatoire**
   1.1. **Chronologie des symptômes** :
   - Question 1 : Date de début des symptômes ?
   - Question 2 : Périodes de répit depuis le début ?
   - Question 3 : Présence d'une saisonnalité ?
   
   1.2. **Nature des symptômes** :
   ✓ **Nez** (rechercher) :
   - Obstruction nasale
   - Écoulement
   - Prurit/Éternuements
   - Respiration buccale
   - Renflements
   ✓ **Œil** (rechercher) :
   - Rougeur/Larmoiement
   - Prurit/Sensation de brûlure
   ✓ **Larynx** :
   - Prurit laryngé/Raclement de gorge ?
   ✓ **Poumons** :
   - Gêne respiratoire (repos/effort/fou rire) ?
   - Respiration sifflante ?
   - Toux (diurne/nocturne) ?
   ✓ **Autres** :
   - Reflux gastro-œsophagien ?
   - Antécédents de traitement :
     * Antihistaminiques (efficacité ?)
     * Ventoline (efficacité ?)
   1.3. **Environnement** :
   ✓ **Logement** :
   - Humidité/Type de sol
   - Animaux domestiques
   - Exposition tabagique
   - État de la literie
   ✓ **Profession** :
   - Exposition professionnelle (ex : boulanger, coiffeur...)
2. **Examen clinique**
   - Signes physiques (ex : cernes chez l'enfant)
   - Examen ORL
   - Auscultation cardio-pulmonaire
3. **Examens para-cliniques**
   - Bilan allergologique (acariens/moisissures/animaux/pollens)
   - EFR si suspicion d'asthme
4. **Traitement**
   - Mesures d'éviction
   - Antihistaminiques/traitements locaux
   - Prise en charge de l'asthme
   - Désensibilisation si indication
"""

def retrieve_and_ask(top_docs, question, conversation_text):
    if not top_docs:
        return "Aucun document trouvé pour répondre à la question."
    
    retrieved_texts = "\n\n".join([doc["chunk_text"] for doc in top_docs])
    # Utiliser agent.ask() au lieu de agent.print_response()
    response = agent.run(
        f"le Logigramme : {logigramme}\n\n"
        f"Documentation : {retrieved_texts}\n\n"
        f"Discussion : {conversation_text}\n\n"
        f"Question : {question}"
    )
    
    return response.content
