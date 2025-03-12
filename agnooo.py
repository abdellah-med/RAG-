from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Création de l'agent au moment où le fichier est importé
if not GROQ_API_KEY:
    raise ValueError("La clé API GROQ_API_KEY n'est pas définie dans les variables d'environnement.")

# L'agent est créé dès que ce fichier est importé
agent = Agent(
    model=Groq(id="qwen-2.5-32b", api_key=GROQ_API_KEY),
    description="Tu es un assistant médical spécialisé en allergologie respiratoire.",
    instructions=[
    "Lis attentivement le Logigramme.",
    "Analyse la discussion entre le médecin et le patient.",
    "Prends en compte la documentation pour mieux comprendre les symptômes et les antécédents.",
    "Propose une seule question pertinente à poser au patient en fonction des informations disponibles."
    ], 
    markdown=True
)

def retrieve_and_ask(top_docs, question):
    if not top_docs:
        return "Aucun document trouvé pour répondre à la question."
    
    retrieved_texts = "\n\n".join([doc["chunk_text"] for doc in top_docs])
    response = agent.print_response(
    f"Logigramme pour le diagnostic des allergies respiratoires\n"
    "L'interrogatoire\n"
    "1.1. Chronologie des symptômes\n"
    "Question 1 : Date de début des symptômes\n"
    "Question 2 : Y a-t-il eu des périodes de répit depuis le début des symptômes ?\n"
    "Question 3 : Y a-t-il une saisonnalité ?\n\n"
    "1.2. Nature des symptômes\n"
    "Le nez : obstruction, écoulement, prurit (démangeaison), éternuements, respiration buccale (manifestation fréquente de l'obstruction nasale), renflements.\n"
    "Les yeux : rougeur, larmoiement, prurit, sensation de brûlure ou de « sable dans les yeux ».\n"
    "Prurit laryngé et/ou raclement de gorge : présence ou non.\n"
    "Les poumons : gêne respiratoire au repos et/ou à l'effort (notamment au fou rire), respiration sifflante, toux (diurne, nocturne ou les deux).\n"
    "Reflux gastro-œsophagien : présence ou non.\n"
    "Traitements utilisés : antihistaminiques (efficacité ?), Ventoline (efficacité ?).\n\n"
    "1.3. Environnement\n"
    "Type de logement : humidité, type de revêtement au sol, présence ou non d'animaux domestiques, exposition tabagique, état de la literie.\n"
    "Profession : exposition professionnelle éventuelle (exemple : boulangers, coiffeurs…)\n\n"
    f"Documentation : {retrieved_texts}\n\n"
    "Discussion :\n"
    "Allergologue : Bonjour, Monsieur. Je suis le Dr. Lambert, allergologue. Que puis-je faire pour vous aujourd'hui ?\n"
    "Patient : Bonjour, docteur. Depuis quelques mois, j'ai souvent le nez bouché et une sensation d'inconfort au niveau de la gorge, surtout le matin.\n\n"
    f"Question : {question}"
    )
     
    
    return response
