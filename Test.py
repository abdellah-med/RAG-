import pymupdf  # PyMuPDF
import re

def generer_chunks_paragraphes(paragraphes, taille_chunk, chevauchement):
    """
    Génère des chunks de texte à partir des paragraphes extraits,
    en comptant le nombre de mots plutôt que le nombre de caractères.
    """
    chunks = []
    mots = " ".join(paragraphes).split()  # Fusionner le texte en une liste de mots
    n = len(mots)

   

    debut = 0
    while debut < n:
        fin = min(debut + taille_chunk, n)  
        chunk = " ".join(mots[debut:fin])  
        chunks.append(chunk)

        # Vérification pour éviter une boucle infinie
        if fin == n:
            break  # Sortir de la boucle si on atteint la fin du texte

        debut += (taille_chunk - chevauchement)  # Avancer en respectant le chevauchement

    return chunks

def lire_pdf_paragraphes(chemin_pdf, taille_chunk, chevauchement):
    """
    Lit un PDF, extrait les paragraphes et génère des chunks en fonction du nombre de mots.
    """
    try:
        doc = pymupdf.open(chemin_pdf)  # Ouvrir le PDF
        contenu_complet = []

        for num_page in range(len(doc)):
            page = doc[num_page]  # Récupérer la page
            texte = page.get_text("text").strip()  # Extraire le texte et nettoyer les espaces

            if texte:  # Ne pas ajouter les pages vides
                contenu_complet.append(texte)

        doc.close()

        # Vérifier si le document contient du texte
        if not contenu_complet:
            print(" Le document est vide ou n'a pas de texte détectable.")
            return

        # Prétraitement : suppression des retours à la ligne multiples et espaces inutiles
        texte_propre = re.sub(r"\n{2,}", "\n", "\n".join(contenu_complet))  # Remplace \n\n par \n
        texte_propre = re.sub(r"\s+", " ", texte_propre).strip()  # Supprime les espaces multiples

        # Découper le texte en paragraphes
        paragraphes = [p.strip() for p in texte_propre.split("\n") if p.strip()]

        # Vérifier qu'il y a bien du texte à traiter
        if not paragraphes:
            print(" Impossible d'extraire des paragraphes valides.")
            return

        # Générer les chunks en fonction des mots
        chunks = generer_chunks_paragraphes(paragraphes, taille_chunk, chevauchement)

        # Affichage des chunks
        print("\n🔹 **Chunks générés avec chevauchement** 🔹\n")
        for i, chunk in enumerate(chunks):
            print(f"🟢 **Chunk {i + 1}**\n{chunk}\n{'-' * 80}\n")

    except Exception as e:
        print(f" Erreur lors de la lecture du PDF : {e}")

# Exemple d'utilisation
chemin_du_pdf = "ALLERG_IA/1988GRE17008_gillet_martine(1)(D)_MP_version_diffusion.pdf"
lire_pdf_paragraphes(chemin_du_pdf, taille_chunk=200, chevauchement=75)
