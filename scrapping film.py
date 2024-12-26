# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:13:16 2024

@author: cleme
"""
import locale
import pandas as pd
import random as rd
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration locale et options
try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error as e:
    print(f"Erreur lors de la configuration de la locale : {e}")

pd.set_option('mode.chained_assignment', None)

chrome_options = Options()
chrome_options.add_argument("--disable-notifications")  # Désactiver les notifications
chrome_options.add_argument("--disable-popup-blocking")  # Désactiver les pop-ups

url = "https://www.allocine.fr/films/"
try:
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
except Exception as e:
    print(f"Erreur lors du lancement du navigateur : {e}")
    exit()

wait = WebDriverWait(driver, 60)

time.sleep(0.5 + rd.random())

# Gestion des cookies
try:
    cookie_buttons = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "jad_cmp_paywall_button"))
    )
    cookie_buttons[1].click()
except Exception as e:
    print(f"Erreur lors de la gestion des cookies : {e}")

time.sleep(0.5 + rd.random())
time.sleep(15)

# Initialisation du DataFrame
df = pd.DataFrame(columns=['titre', 'Presse', 'Spectateurs', 'date', 'Realisateur', 'genre', 'acteur', 'durée','page'])

# Boucle principale
for i in range(8120):
    try:
        url = f"https://www.allocine.fr/films/?page={i + 1}"
        driver.get(url)
    except Exception as e:
        print(f"Erreur lors de l'accès à la page {url} : {e}")
        continue

    try:
        li_elements = driver.find_elements(By.CSS_SELECTOR, "li.mdl")
        links = [
            li.find_element(By.CLASS_NAME, "meta-title-link").get_attribute("href")
            for li in li_elements if 'Spectateurs' in li.text
        ]
    except Exception as e:
        print(f"Erreur lors de la collecte des liens : {e}")
        continue

    # Parcours des liens
    for link in links:
        try:
            driver.get(link)
            new_row = pd.DataFrame([[None] * len(df.columns)], columns=list(df.columns))
            df = pd.concat([df, new_row], ignore_index=True)

            try:
                df['titre'].iloc[-1] = driver.find_element(By.CLASS_NAME, "titlebar.titlebar-page").text[:-3]
                df['page'].iloc[-1] = i
            except Exception as e:
                print(f"Erreur lors de l'extraction du titre : {e}")

            # Récupérer les notes
            try:
                ratings = driver.find_elements(By.CLASS_NAME, "rating-item")
                for rating in ratings:
                    if 'Presse' in rating.text:
                        presse = rating.find_element(By.CLASS_NAME, "stareval-note").text
                        df['Presse'].iloc[-1] = float(presse.replace(',', '.'))
                    if 'Spectateurs' in rating.text:
                        spectateur = rating.find_element(By.CLASS_NAME, "stareval-note").text
                        df['Spectateurs'].iloc[-1] = float(spectateur.replace(',', '.'))
            except Exception as e:
                print(f"Erreur lors de l'extraction des notes : {e}")

            # Informations sur le film
            try:
                carte_film = driver.find_element(By.CSS_SELECTOR, ".card.entity-card.entity-card-list.cf.entity-card-player-ovw")
                ligne = carte_film.find_element(By.CSS_SELECTOR, ".meta-body-item.meta-body-info").text.split('|')
                if len(ligne) > 0:
                    df['date'].iloc[-1] = ligne[0].strip()
                if len(ligne) > 1:
                    df['durée'].iloc[-1] = ligne[1].strip()
                if len(ligne) > 2:
                    df['genre'].iloc[-1] = ligne[2].strip()

                try:
                    df['Realisateur'].iloc[-1] = carte_film.find_element(By.CSS_SELECTOR, ".meta-body-item.meta-body-direction.meta-body-oneline").text
                except Exception:
                    df['Realisateur'].iloc[-1] = None

                try:
                    df['acteur'].iloc[-1] = carte_film.find_element(By.CSS_SELECTOR, ".meta-body-item.meta-body-actor").text
                except Exception:
                    df['acteur'].iloc[-1] = None
            except Exception as e:
                print(f"Erreur lors de l'extraction des informations du film : {e}")

            # Récupérer des données supplémentaires si disponibles
            try:
                element = driver.find_element(By.CSS_SELECTOR, ".section.ovw.ovw-technical")
                infos = element.find_elements(By.CLASS_NAME, "item")
                for info in infos:
                    column = info.find_element(By.CLASS_NAME, "what.light").text
                    data = info.find_element(By.CSS_SELECTOR, "[class*='that']").text
                    if column in list(df.columns):
                        df[column].iloc[-1] = data
                    else:
                        df[column] = [None] * len(df)
                        df[column].iloc[-1] = data
            except Exception as e:
                print(f"Erreur lors de l'extraction des informations supplémentaires : {e}")

        except Exception as e:
            print(f"Erreur lors de l'accès au lien {link} : {e}")
            continue

    # Sauvegarde périodique
    try:
        df.to_excel("C:/Users/cleme/Desktop/scrapping_film.xlsx", index=False)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier Excel pour la page {i + 1} : {e}")

# Fermeture du navigateur
try:
    driver.quit()
except Exception as e:
    print(f"Erreur lors de la fermeture du navigateur : {e}")

