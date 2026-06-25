import os
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIGURARE FOLDER ȘI DIMENSIUNI
# ==========================================
FOLDER_POZE = "imagini_custom"  # Schimbă dacă folderul tău se numește altfel
NUME_FISIER_FINAL = "Plansa_Thresholds_Custom.png"
TITLU_PRINCIPAL = "Thresholds Custom "

# Definim structura pe rânduri (Titlu pe mijloc, poza stânga, poza dreapta)
MODELE = [
    {
        "titlu": "Evaluare Fusion\nEnd-to-End",
        "metrice": "metrice_end2end.jpeg",
        "matrice": "matrice_end2end.jpeg"
    },
    {
        "titlu": "Evaluare Fusion\nVectori (Offline)",
        "metrice": "metrice_vectori.jpeg",
        "matrice": "matrice_vectori.jpeg"
    },
    {
        "titlu": "Evaluare CNN\n(Imagine)",
        "metrice": "metrice_cnn.jpeg",
        "matrice": "matrice_cnn.jpeg"
    },
    {
        "titlu": "Evaluare BERT\n(Text)",
        "metrice": "metrice_bert.jpeg",
        "matrice": "matrice_bert.jpeg"
    }
]

# Dimensiunile pânzei (Canvas) - Rezoluție foarte mare pentru claritate
LATIME_CANVAS = 4000
INALTIME_RAND = 1000
INALTIME_HEADER = 300
INALTIME_TOTALA = INALTIME_HEADER + (len(MODELE) * INALTIME_RAND)

# Dimensiuni Coloane
LATIME_STANGA = 1700  # Pentru tabelul de metrici (care e mai lat)
LATIME_MIJLOC = 600  # Pentru textul cu numele evaluării
LATIME_DREAPTA = 1700  # Pentru matricele de confuzie (care sunt pătrate)


# ==========================================
# FUNCȚII UTILITARE
# ==========================================
def redimensioneaza_si_pastreaza_proportii(img, max_width, max_height):
    """Redimensionează o poză ca să încapă în cutia ei, păstrând aspect ratio-ul."""
    ratio = min(max_width / img.width, max_height / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    # Folosim Resampling.LANCZOS pentru a menține textul foarte clar
    return img.resize(new_size, Image.Resampling.LANCZOS)


# ==========================================
# EXECUȚIA PRINCIPALĂ
# ==========================================
if __name__ == "__main__":
    print("Se pregătește pânza (Canvas-ul)...")
    # Creăm o imagine complet albă
    canvas = Image.new('RGB', (LATIME_CANVAS, INALTIME_TOTALA), color='white')
    draw = ImageDraw.Draw(canvas)

    # Încercăm să încărcăm fonturi Windows implicite (Arial sau Tahoma)
    try:
        font_titlu = ImageFont.truetype("arialbd.ttf", 120)  # Arial Bold
        font_modele = ImageFont.truetype("arialbd.ttf", 70)
    except IOError:
        print("Fontul Arial nu a fost găsit. Se folosește fontul default...")
        font_titlu = ImageFont.load_default()
        font_modele = ImageFont.load_default()

    # Desenăm Titlul Principal sus, pe mijloc
    bbox = draw.textbbox((0, 0), TITLU_PRINCIPAL, font=font_titlu)
    latime_text = bbox[2] - bbox[0]
    x_titlu = (LATIME_CANVAS - latime_text) // 2
    y_titlu = 80
    draw.text((x_titlu, y_titlu), TITLU_PRINCIPAL, fill="black", font=font_titlu)

    # Trasăm o linie despărțitoare sub titlu
    draw.line([(200, INALTIME_HEADER - 20), (LATIME_CANVAS - 200, INALTIME_HEADER - 20)], fill="black", width=5)

    print("Se procesează și se aliniază imaginile...")

    # Iterăm prin fiecare model (fiecare rând)
    for index, model in enumerate(MODELE):
        y_curent = INALTIME_HEADER + (index * INALTIME_RAND)

        # 1. TEXTUL DE PE MIJLOC
        bbox_text = draw.textbbox((0, 0), model['titlu'], font=font_modele)
        latime_text = bbox_text[2] - bbox_text[0]
        inaltime_text = bbox_text[3] - bbox_text[1]

        x_text = LATIME_STANGA + (LATIME_MIJLOC - latime_text) // 2
        y_text = y_curent + (INALTIME_RAND - inaltime_text) // 2

        # Desenăm textul (centrat)
        draw.multiline_text((x_text, y_text), model['titlu'], fill="black", font=font_modele, align="center")

        # 2. POZA DIN STÂNGA (METRICE)
        cale_metrice = os.path.join(FOLDER_POZE, model['metrice'])
        if os.path.exists(cale_metrice):
            img_metrice = Image.open(cale_metrice)
            # Lăsăm un mic padding (margine) de 50px
            img_metrice_resized = redimensioneaza_si_pastreaza_proportii(img_metrice, LATIME_STANGA - 100,
                                                                         INALTIME_RAND - 100)

            # Centram poza in cutia din stanga
            x_img_stanga = 50 + (LATIME_STANGA - 100 - img_metrice_resized.width) // 2
            y_img_stanga = y_curent + 50 + (INALTIME_RAND - 100 - img_metrice_resized.height) // 2

            canvas.paste(img_metrice_resized, (x_img_stanga, y_img_stanga))
        else:
            print(f"  [Avertisment] Nu am găsit {cale_metrice}")

        # 3. POZA DIN DREAPTA (MATRICE)
        cale_matrice = os.path.join(FOLDER_POZE, model['matrice'])
        if os.path.exists(cale_matrice):
            img_matrice = Image.open(cale_matrice)
            img_matrice_resized = redimensioneaza_si_pastreaza_proportii(img_matrice, LATIME_DREAPTA - 100,
                                                                         INALTIME_RAND - 100)

            # Centram poza in cutia din dreapta
            x_img_dreapta = LATIME_STANGA + LATIME_MIJLOC + 50 + (LATIME_DREAPTA - 100 - img_matrice_resized.width) // 2
            y_img_dreapta = y_curent + 50 + (INALTIME_RAND - 100 - img_matrice_resized.height) // 2

            canvas.paste(img_matrice_resized, (x_img_dreapta, y_img_dreapta))
        else:
            print(f"  [Avertisment] Nu am găsit {cale_matrice}")

        # Trasăm o linie fină despărțitoare între rânduri (mai puțin la ultimul)
        if index < len(MODELE) - 1:
            draw.line([(100, y_curent + INALTIME_RAND), (LATIME_CANVAS - 100, y_curent + INALTIME_RAND)],
                      fill="lightgray", width=3)

    # Salvăm rezultatul
    canvas.save(NUME_FISIER_FINAL)
    print(f"\n✅ Planșa a fost generată cu succes! Deschide fișierul: {NUME_FISIER_FINAL}")