import pymupdf  # PyMuPDF

def generer_chunks_paragraphes(paragraphes, taille_chunk=100, chevauchement=20):
    """
    Génère des chunks de texte à partir des paragraphes extraits, 
    en comptant le nombre de mots plutôt que le nombre de caractères.
    """
    chunks = []
    mots = " ".join(paragraphes).split()  # Fusionner le texte en une liste de mots
    debut = 0

    while debut < len(mots):
        fin = min(debut + taille_chunk, len(mots))
        chunk = " ".join(mots[debut:fin])  # Recomposer le chunk sous forme de texte
        chunks.append(chunk)
        debut = fin - chevauchement  # Appliquer le chevauchement

    return chunks

def lire_pdf_paragraphes(chemin_pdf, taille_chunk=100, chevauchement=20):
    """
    Lit un PDF, extrait les paragraphes et génère des chunks en fonction du nombre de mots.
    """
    try:
        doc = pymupdf.open(chemin_pdf)  # Ouvrir le PDF
        contenu_complet = ""

        for num_page in range(len(doc)):
            page = doc[num_page]  # Récupérer la page
            texte = page.get_text("text")  # Extraire le texte
            contenu_complet += texte + "\n"  # Ajouter le texte au contenu global
        
        doc.close()

        # Découper le texte en paragraphes (en détectant les doubles sauts de ligne)
        paragraphes = [p.strip() for p in contenu_complet.split("\n\n") if p.strip()]

        # Générer et afficher les chunks basés sur le nombre de mots
        chunks = generer_chunks_paragraphes(paragraphes, taille_chunk, chevauchement)
        
        print("\n--- Chunks Overlapping ---\n")
        for i, chunk in enumerate(chunks):
            print(f"[Chunk {i + 1}]\n{chunk}\n\n{'-' * 40}\n")

    except Exception as e:
        print(f"Erreur lors de la lecture du PDF : {e}")

# Exemple d'utilisation
chemin_du_pdf = "ALLERG_IA/Bosse-Fortjacques-Romay_Vertigo-39895.pdf"
lire_pdf_paragraphes(chemin_du_pdf, taille_chunk=100, chevauchement=20)
